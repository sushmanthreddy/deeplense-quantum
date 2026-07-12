import unittest

import torch
import numpy as np

from d4_orqb.analyze import (
    mcnemar_exact,
    paired_bootstrap_accuracy,
    probability_ensemble_logits,
)
from d4_orqb.data import (
    hash_ranked_subset,
    index_membership_sha256,
    stratified_hash_folds,
)
from d4_orqb.model import (
    HAAR_MORPHOLOGY_CONTEXT_INDICES,
    D4OrbitClassifier,
    PhysicsChannelBank,
    annular_haar_scattering_summary,
    d4_transform,
    d4_views,
    invariant_annular_haar_coefficients,
    lens_morphology_summary,
    paired_spatial_statistics,
    spectral_morphology_summary,
)
from d4_orqb.lenspinn import ScalarLensInversion, ShiftPatchTokenizer, lenspinn_distortion
from d4_orqb.quantum import (
    CAYLEY_EDGE_FAMILIES,
    D4_ELEMENTS,
    D4_INDEX,
    D4OrbitQuantumBottleneck,
    R2S_PLAQUETTES,
    R2S_EDGES,
    R2_EDGES,
    RS_PLAQUETTES,
    RS_EDGES,
    R3S_EDGES,
    R_EDGES,
    S_EDGES,
    d4_multiply,
    right_regular_permutation,
)
from d4_orqb.train import (
    clone_module_state,
    fit_haar_morphology_normalization,
    fit_haar_subtype_selection,
    haar_subtype_exact_replay,
    hierarchical_model_i_loss,
    knowledge_distillation_loss,
    correctness_gated_oof_distillation_loss,
    morphology_path_sensitivity_order,
    prefix_slice_output_tensor,
    physics_augment_batch,
    remap_projection_encoder_and_summary,
    remap_projection_to_multiscale_correlation,
    remap_morphology_kd_to_haar_candidate,
    remap_haar_to_subtype_residual,
    remap_haar_to_shared_late_refinement,
    restore_fresh_haar_core,
    select_model_i_subtype_task,
    select_haar_subtype_coefficients,
    soft_target_cross_entropy,
    shared_late_refinement_exact_replay,
    shared_late_refinement_initialization_record,
    shape_compatible_prefix_state,
    subtype_mixup_batch,
    tied_mean_dispersion_initialization_record,
    zero_extend_input_weight,
)


class D4OrbitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_orbit_lift_is_right_regular(self):
        image = torch.arange(16.0).reshape(1, 1, 4, 4)
        base = d4_views(image)
        for element in D4_ELEMENTS:
            transformed = d4_views(d4_transform(image, *element))
            expected = base.index_select(1, right_regular_permutation(element))
            torch.testing.assert_close(transformed, expected, rtol=0.0, atol=0.0)

    def test_quantum_core_covariance_and_invariance(self):
        core = D4OrbitQuantumBottleneck(heads=2, reuploads=2)
        features = torch.randn(3, 2, 2, 8)
        base_invariant, base_equivariant = core(features, return_equivariant=True)
        for element in D4_ELEMENTS:
            permutation = right_regular_permutation(element)
            moved = features.index_select(-1, permutation)
            invariant, equivariant = core(moved, return_equivariant=True)
            torch.testing.assert_close(invariant, base_invariant, rtol=2e-5, atol=2e-5)
            for name in ("z", "x"):
                expected = base_equivariant[name].index_select(-1, permutation)
                torch.testing.assert_close(equivariant[name], expected, rtol=2e-5, atol=2e-5)

    def test_cayley_edge_orbits_and_quantum_parameter_count(self):
        edge_families = ((R_EDGES, 8), (R2_EDGES, 4), (S_EDGES, 4))
        for edges, expected_size in edge_families:
            edge_set = {tuple(sorted(edge)) for edge in edges}
            self.assertEqual(len(edge_set), expected_size)
            for element in D4_ELEMENTS:
                moved = {
                    tuple(
                        sorted(
                            (
                                D4_INDEX[d4_multiply(D4_ELEMENTS[a], element)],
                                D4_INDEX[d4_multiply(D4_ELEMENTS[b], element)],
                            )
                        )
                    )
                    for a, b in edge_set
                }
                self.assertEqual(moved, edge_set)

        core = D4OrbitQuantumBottleneck(heads=4, reuploads=2)
        self.assertEqual(core.params.shape, (4, 2, 11))
        self.assertEqual(core.parameter_report()["quantum_trainable"], 88)

    def test_full_model_is_invariant_and_quantum_trains(self):
        model = D4OrbitClassifier(heads=2, reuploads=2)
        image = torch.rand(2, 1, 32, 32)
        model.eval()
        reference = model(image)
        for element in D4_ELEMENTS:
            actual = model(d4_transform(image, *element))
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
        model.train()
        loss = model(image).square().mean()
        loss.backward()
        self.assertIsNotNone(model.core.params.grad)
        self.assertGreater(float(model.core.params.grad.norm()), 0.0)
        self.assertLess(model.parameter_report()["total"], 300_000)

    def test_small_encoder_stays_below_one_million_parameters(self):
        model = D4OrbitClassifier(heads=2, reuploads=2, encoder_variant="small")
        report = model.parameter_report()
        self.assertLess(report["total"], 1_000_000)
        self.assertEqual(report["encoder_variant"], "small")

    def test_half_budget_encoders_physics_moments_and_gibbs_core(self):
        half_ceiling = 245_221 // 2
        expected = {
            ("micro", "angle"): 122_437,
            ("micro", "gibbs"): 122_485,
            ("eca", "angle"): 121_899,
            ("eca", "gibbs"): 121_947,
        }
        image = torch.rand(1, 1, 32, 32)
        for (encoder, encoding), parameters in expected.items():
            model = D4OrbitClassifier(
                encoder_variant=encoder,
                physics_summary="moments",
                quantum_encoding=encoding,
                dropout=0.0,
            )
            report = model.parameter_report()
            self.assertEqual(report["total"], parameters)
            self.assertLessEqual(report["total"], half_ceiling)
            self.assertEqual(report["physics_summary_dim"], 16)
            self.assertEqual(model.orbit_projection.in_features, model.encoder.output_dim + 16)
            self.assertEqual(sum(p.numel() for p in model.physics.parameters()), 0)
            logits, auxiliary = model(image, return_aux=True)
            self.assertEqual(tuple(logits.shape), (1, 3))
            self.assertEqual(tuple(auxiliary["angles"].shape), (1, 4, 2, 8))
            self.assertEqual(tuple(auxiliary["invariants"].shape), (1, 48))
            self.assertTrue(torch.isfinite(logits).all())

        statistical = D4OrbitClassifier(
            encoder_variant="micro-stat",
            physics_summary="moments",
            reuploads=3,
            dropout=0.0,
        )
        statistical_report = statistical.parameter_report()
        self.assertEqual(statistical_report["total"], 122_573)
        self.assertEqual(statistical_report["quantum"], 132)
        self.assertLessEqual(statistical_report["total"], half_ceiling)
        self.assertEqual(statistical.encoder.output_dim, 192)
        self.assertEqual(statistical.orbit_projection.in_features, 208)
        self.assertEqual(statistical.head[1].out_features, 19)
        statistical.eval()
        reference = statistical(image)
        for element in D4_ELEMENTS:
            actual = statistical(d4_transform(image, *element))
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)

        deep = D4OrbitClassifier(
            encoder_variant="deep-se",
            physics_summary="moments",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        deep_report = deep.parameter_report()
        self.assertEqual(deep_report["total"], 122_589)
        self.assertEqual(deep_report["encoder"], 119_002)
        self.assertEqual(deep_report["orbit_projection"], 1_736)
        self.assertEqual(deep_report["quantum"], 88)
        self.assertEqual(deep_report["head_and_context"], 1_763)
        self.assertLessEqual(deep_report["total"], half_ceiling)
        deep.eval()
        deep_reference = deep(image)
        for element in D4_ELEMENTS:
            deep_actual = deep(d4_transform(image, *element))
            torch.testing.assert_close(
                deep_actual, deep_reference, rtol=3e-5, atol=3e-5
            )

        micro_source = D4OrbitClassifier(
            encoder_variant="micro", physics_summary="moments"
        )
        prefixes = (
            "encoder.stem.",
            "encoder.blocks.0.",
            "encoder.blocks.1.",
            "encoder.blocks.2.",
            "encoder.blocks.3.",
            "encoder.blocks.4.",
            "orbit_projection.",
            "core.",
            "head.",
        )
        compatible, skipped = shape_compatible_prefix_state(
            micro_source.state_dict(), deep.state_dict(), prefixes
        )
        self.assertGreaterEqual(len(compatible), 70)
        self.assertEqual([item["key"] for item in skipped], ["orbit_projection.weight"])
        self.assertTrue(all(key.startswith(prefixes) for key in compatible))
        compatible["orbit_projection.weight"] = micro_source.state_dict()[
            "orbit_projection.weight"
        ]
        adapted = []
        zero_extend_input_weight(
            compatible,
            deep.state_dict(),
            "orbit_projection.weight",
            adapted,
            insert_before_tail=16,
        )
        self.assertEqual(adapted[0]["source_shape"], (8, 112))
        self.assertEqual(adapted[0]["target_shape"], (8, 216))
        remapped = compatible["orbit_projection.weight"]
        source_projection = micro_source.orbit_projection.weight.detach()
        torch.testing.assert_close(remapped[:, :96], source_projection[:, :96])
        torch.testing.assert_close(remapped[:, 96:200], torch.zeros(8, 104))
        torch.testing.assert_close(remapped[:, 200:], source_projection[:, 96:])
        incompatible = deep.load_state_dict(compatible, strict=False)
        self.assertFalse(incompatible.unexpected_keys)

        morphology = D4OrbitClassifier(
            encoder_variant="deep-se-morph",
            physics_summary="moments-morphology",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        morphology_report = morphology.parameter_report()
        self.assertEqual(morphology_report["total"], 122_573)
        self.assertEqual(morphology_report["encoder"], 118_218)
        self.assertEqual(morphology_report["orbit_projection"], 2_152)
        self.assertEqual(morphology_report["quantum"], 88)
        self.assertEqual(morphology_report["head_and_context"], 2_115)
        self.assertEqual(morphology_report["physics_summary_dim"], 76)
        self.assertEqual(morphology_report["morphology_context_dim"], 60)
        self.assertLessEqual(morphology_report["total"], half_ceiling)
        self.assertEqual(morphology.encoder.output_dim, 192)
        self.assertEqual(morphology.orbit_projection.in_features, 268)
        self.assertEqual(morphology.head.invariant_norm.normalized_shape, (48,))
        self.assertEqual(morphology.head.projection.in_features, 108)
        self.assertEqual(morphology.head.projection.out_features, 18)
        morphology.eval()
        morphology_logits, morphology_auxiliary = morphology(
            image, return_aux=True
        )
        self.assertEqual(tuple(morphology_logits.shape), (1, 3))
        self.assertEqual(
            tuple(morphology_auxiliary["morphology_context"].shape), (1, 60)
        )
        morphology_reference = morphology_logits
        for element in D4_ELEMENTS:
            morphology_actual = morphology(d4_transform(image, *element))
            torch.testing.assert_close(
                morphology_actual,
                morphology_reference,
                rtol=4e-5,
                atol=4e-5,
            )

        source_state = deep.state_dict()
        target_state = morphology.state_dict()
        adapted_state = {
            key: source_state[key].detach().clone()
            for key in (
                "encoder.final.0.weight",
                "encoder.final.1.weight",
                "encoder.final.1.bias",
                "orbit_projection.weight",
            )
        }
        adapted_tensors = []
        for key in (
            "encoder.final.0.weight",
            "encoder.final.1.weight",
            "encoder.final.1.bias",
        ):
            prefix_slice_output_tensor(
                adapted_state, target_state, key, adapted_tensors
            )
        remap_projection_encoder_and_summary(
            adapted_state,
            target_state,
            "orbit_projection.weight",
            adapted_tensors,
            source_encoder_dim=200,
            target_encoder_dim=192,
            preserved_summary_dim=16,
        )
        self.assertEqual(tuple(adapted_state["encoder.final.0.weight"].shape), (192, 96, 1, 1))
        torch.testing.assert_close(
            adapted_state["encoder.final.0.weight"],
            source_state["encoder.final.0.weight"][:192],
        )
        remapped_projection = adapted_state["orbit_projection.weight"]
        torch.testing.assert_close(
            remapped_projection[:, :192],
            source_state["orbit_projection.weight"][:, :192],
        )
        torch.testing.assert_close(
            remapped_projection[:, 192:208],
            source_state["orbit_projection.weight"][:, 200:216],
        )
        torch.testing.assert_close(
            remapped_projection[:, 208:], torch.zeros(8, 60)
        )

        gibbs = D4OrbitClassifier(
            encoder_variant="eca",
            physics_summary="moments",
            quantum_encoding="gibbs",
            dropout=0.0,
        )
        gibbs.eval()
        reference = gibbs(image)
        for element in D4_ELEMENTS:
            actual = gibbs(d4_transform(image, *element))
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)
        gibbs.train()
        gibbs(image).square().mean().backward()
        self.assertIsNotNone(gibbs.core.energy_params.grad)
        self.assertGreater(float(gibbs.core.energy_params.grad.norm()), 0.0)

    def test_multiscale_correlation_quantum_clean_budget_gradients_and_remap(self):
        pattern = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        feature_map = torch.stack(
            (pattern, 3.0 * pattern, 2.0 + pattern, -2.0 * pattern), dim=0
        ).unsqueeze(0)
        feature_map.requires_grad_(True)
        statistics = paired_spatial_statistics(feature_map)
        self.assertEqual(tuple(statistics.shape), (1, 10))
        torch.testing.assert_close(
            statistics[:, -2:],
            torch.tensor([[1.0, -1.0]]),
            rtol=2e-5,
            atol=2e-5,
        )
        statistics.square().sum().backward()
        self.assertIsNotNone(feature_map.grad)
        self.assertTrue(torch.isfinite(feature_map.grad).all())
        self.assertGreater(float(feature_map.grad.norm()), 0.0)

        half_ceiling = 245_221 // 2
        model = D4OrbitClassifier(
            encoder_variant="deep-se-mscorr",
            physics_summary="moments",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        report = model.parameter_report()
        self.assertEqual(report["total"], 122_549)
        self.assertEqual(report["encoder"], 117_042)
        self.assertEqual(report["orbit_projection"], 3_656)
        self.assertEqual(report["quantum"], 88)
        self.assertEqual(report["head_and_context"], 1_763)
        self.assertEqual(report["encoder_output_dim"], 440)
        self.assertEqual(report["encoder_final_channels"], 180)
        self.assertTrue(report["encoder_multiscale_correlation_pool"])
        self.assertEqual(report["physics_summary_dim"], 16)
        self.assertEqual(report["morphology_context_dim"], 0)
        self.assertLessEqual(report["total"], half_ceiling)
        self.assertEqual(model.encoder.output_dim, 440)
        self.assertEqual(model.encoder.final[0].out_channels, 180)
        self.assertEqual(model.orbit_projection.in_features, 456)
        self.assertEqual(model.head[0].normalized_shape, (48,))
        self.assertEqual(model.head[1].in_features, 48)

        tapped_shapes = {}

        def capture(name):
            def hook(_module, _inputs, output):
                tapped_shapes[name] = tuple(output.shape)

            return hook

        handles = (
            model.encoder.blocks[3].register_forward_hook(capture("block3")),
            model.encoder.blocks[5].register_forward_hook(capture("block5")),
            model.encoder.final.register_forward_hook(capture("final")),
        )
        encoder_input = torch.rand(2, 8, 96, 96, requires_grad=True)
        descriptor = model.encoder(encoder_input)
        for handle in handles:
            handle.remove()
        self.assertEqual(tapped_shapes["block3"], (2, 40, 12, 12))
        self.assertEqual(tapped_shapes["block5"], (2, 64, 6, 6))
        self.assertEqual(tapped_shapes["final"], (2, 180, 3, 3))
        self.assertEqual(tuple(descriptor.shape), (2, 440))
        self.assertTrue(torch.isfinite(descriptor).all())
        descriptor.square().mean().backward()
        self.assertIsNotNone(encoder_input.grad)
        self.assertTrue(torch.isfinite(encoder_input.grad).all())
        self.assertGreater(float(encoder_input.grad.norm()), 0.0)

        with self.assertRaisesRegex(ValueError, "context bypass"):
            D4OrbitClassifier(
                encoder_variant="deep-se-mscorr",
                physics_summary="moments",
                include_context=True,
            )
        with self.assertRaisesRegex(ValueError, "post-core morphology"):
            D4OrbitClassifier(
                encoder_variant="deep-se-mscorr",
                physics_summary="moments-morphology",
            )

        model.zero_grad(set_to_none=True)
        image = torch.rand(1, 1, 32, 32, requires_grad=True)
        model.train()
        logits, auxiliary = model(image, return_aux=True)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(auxiliary["encoded"].shape), (1, 8, 440))
        self.assertEqual(tuple(auxiliary["angles"].shape), (1, 4, 2, 8))
        self.assertEqual(tuple(auxiliary["invariants"].shape), (1, 48))
        self.assertIsNone(auxiliary["morphology_context"])
        logits.square().mean().backward()
        self.assertIsNotNone(model.orbit_projection.weight.grad)
        self.assertGreater(float(model.orbit_projection.weight.grad.norm()), 0.0)
        self.assertIsNotNone(model.core.params.grad)
        self.assertGreater(float(model.core.params.grad.norm()), 0.0)
        for block_index in (3, 5):
            gradients = [
                parameter.grad
                for parameter in model.encoder.blocks[block_index].parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.float().square().sum()) for gradient in gradients),
                0.0,
            )

        model.eval()
        image = image.detach()
        reference = model(image)
        for element in D4_ELEMENTS:
            actual = model(d4_transform(image, *element))
            torch.testing.assert_close(actual, reference, rtol=4e-5, atol=4e-5)

        source = D4OrbitClassifier(
            encoder_variant="deep-se",
            physics_summary="moments",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        target = D4OrbitClassifier(
            encoder_variant="deep-se-mscorr",
            physics_summary="moments",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        prefixes = (
            "encoder.stem.",
            "encoder.blocks.",
            "encoder.final.",
            "orbit_projection.",
            "core.",
            "head.",
        )
        source_state = source.state_dict()
        target_state = target.state_dict()
        compatible, _ = shape_compatible_prefix_state(
            source_state, target_state, prefixes
        )
        self.assertIn("core.params", compatible)
        self.assertIn("head.1.weight", compatible)
        self.assertIn("encoder.blocks.7.block.0.weight", compatible)
        adaptation_state = {
            key: source_state[key].detach().clone()
            for key in (
                "encoder.final.0.weight",
                "encoder.final.1.weight",
                "encoder.final.1.bias",
                "orbit_projection.weight",
            )
        }
        adapted = []
        for key in (
            "encoder.final.0.weight",
            "encoder.final.1.weight",
            "encoder.final.1.bias",
        ):
            prefix_slice_output_tensor(
                adaptation_state, target_state, key, adapted
            )
        remap_projection_to_multiscale_correlation(
            adaptation_state,
            target_state,
            "orbit_projection.weight",
            adapted,
            source_encoder_dim=200,
            target_multiscale_dim=260,
            target_final_dim=180,
            preserved_summary_dim=16,
        )
        self.assertEqual(
            tuple(adaptation_state["encoder.final.0.weight"].shape),
            (180, 96, 1, 1),
        )
        torch.testing.assert_close(
            adaptation_state["encoder.final.0.weight"],
            source_state["encoder.final.0.weight"][:180],
        )
        remapped = adaptation_state["orbit_projection.weight"]
        self.assertEqual(tuple(remapped.shape), (8, 456))
        torch.testing.assert_close(remapped[:, :260], torch.zeros(8, 260))
        torch.testing.assert_close(
            remapped[:, 260:440],
            source_state["orbit_projection.weight"][:, :180],
        )
        torch.testing.assert_close(
            remapped[:, 440:],
            source_state["orbit_projection.weight"][:, 200:],
        )
        self.assertEqual(
            adapted[-1]["method"],
            "zero-new-multiscale-copy-final-prefix-and-semantic-summary",
        )
        compatible.update(adaptation_state)
        incompatible = target.load_state_dict(compatible, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertFalse(incompatible.missing_keys)
        torch.testing.assert_close(target.core.params, source.core.params)
        torch.testing.assert_close(target.head[1].weight, source.head[1].weight)
        torch.testing.assert_close(
            target.encoder.blocks[7].state_dict()["block.0.weight"],
            source.encoder.blocks[7].state_dict()["block.0.weight"],
        )

    def test_plaquette_readout_and_parallel_cores_stay_below_half_budget(self):
        expected_rs = {
            (0, 1, 4, 5),
            (0, 3, 5, 6),
            (1, 2, 4, 7),
            (2, 3, 6, 7),
        }
        expected_r2s = {(0, 2, 4, 6), (1, 3, 5, 7)}
        self.assertEqual(set(RS_PLAQUETTES), expected_rs)
        self.assertEqual(set(R2S_PLAQUETTES), expected_r2s)
        for motifs in (RS_PLAQUETTES, R2S_PLAQUETTES):
            for element in D4_ELEMENTS:
                permutation = right_regular_permutation(element).tolist()
                moved = {
                    tuple(sorted(permutation[index] for index in motif))
                    for motif in motifs
                }
                self.assertEqual(moved, set(motifs))

        half_ceiling = 245_221 // 2
        configurations = (
            ("quantum", "angle", "plaquette", 122_437),
            ("classical", "angle", "plaquette", 122_437),
            ("quantum", "angle", "cayley-complete", 122_525),
            ("classical", "angle", "cayley-complete", 122_525),
            ("hybrid", "angle", "pair", 122_526),
            ("hybrid", "angle", "plaquette", 122_526),
            ("hybrid", "gibbs", "plaquette", 122_574),
            ("classical-fusion", "angle", "plaquette", 122_526),
        )
        for core, encoding, readout, expected_parameters in configurations:
            model = D4OrbitClassifier(
                core=core,
                encoder_variant="micro",
                physics_summary="moments",
                quantum_encoding=encoding,
                observable_readout=readout,
                dropout=0.0,
            )
            report = model.parameter_report()
            self.assertEqual(report["total"], expected_parameters)
            self.assertLessEqual(report["total"], half_ceiling)
            if readout == "plaquette":
                self.assertEqual(model.core.output_dim, 64)
                self.assertEqual(model.head[1].in_features, 64)
                self.assertEqual(model.head[1].out_features, 24)
            if readout == "cayley-complete":
                self.assertEqual(model.core.output_dim, 112)
                self.assertEqual(model.head[1].in_features, 112)
                self.assertEqual(model.head[1].out_features, 14)

        self.assertEqual(
            CAYLEY_EDGE_FAMILIES,
            (R_EDGES, R2_EDGES, S_EDGES, RS_EDGES, R2S_EDGES, R3S_EDGES),
        )
        for family_index, edges in enumerate(CAYLEY_EDGE_FAMILIES):
            self.assertEqual(len(edges), 8 if family_index == 0 else 4)
            edge_set = {tuple(sorted(edge)) for edge in edges}
            for element in D4_ELEMENTS:
                permutation = right_regular_permutation(element).tolist()
                moved = {
                    tuple(sorted((permutation[a], permutation[b])))
                    for a, b in edge_set
                }
                self.assertEqual(moved, edge_set)

        image = torch.rand(1, 1, 32, 32)
        plaquette = D4OrbitClassifier(
            encoder_variant="micro",
            physics_summary="moments",
            observable_readout="plaquette",
            dropout=0.0,
        )
        plaquette.eval()
        reference = plaquette(image)
        for element in D4_ELEMENTS:
            actual = plaquette(d4_transform(image, *element))
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)

        complete = D4OrbitClassifier(
            encoder_variant="micro",
            physics_summary="moments",
            observable_readout="cayley-complete",
            dropout=0.0,
        )
        complete.eval()
        complete_reference = complete(image)
        for element in D4_ELEMENTS:
            complete_actual = complete(d4_transform(image, *element))
            torch.testing.assert_close(
                complete_actual, complete_reference, rtol=3e-5, atol=3e-5
            )

        hybrid = D4OrbitClassifier(
            core="hybrid",
            encoder_variant="micro",
            physics_summary="moments",
            observable_readout="plaquette",
            dropout=0.0,
        )
        hybrid.train()
        logits, auxiliary = hybrid(image, return_aux=True)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(auxiliary["branch_logits"][0].shape), (1, 3))
        self.assertEqual(tuple(auxiliary["branch_logits"][1].shape), (1, 3))
        self.assertAlmostEqual(float(auxiliary["mixing_weight"]), 0.5, places=7)
        (
            logits.square().mean()
            + 0.1
            * sum(branch.square().mean() for branch in auxiliary["branch_logits"])
        ).backward()
        self.assertGreater(float(hybrid.core.branch_a.params.grad.norm()), 0.0)
        self.assertGreater(float(hybrid.core.branch_b.params.grad.norm()), 0.0)
        self.assertGreater(float(hybrid.core.mix_logit.grad.abs()), 0.0)

    def test_spectral_morphology_summary_budget_invariance_and_initialization(self):
        image = torch.rand(2, 1, 96, 96)
        bank = PhysicsChannelBank()
        summary = spectral_morphology_summary(bank(image))
        self.assertEqual(tuple(summary.shape), (2, 16))
        self.assertTrue(torch.isfinite(summary).all())
        for element in D4_ELEMENTS:
            transformed = spectral_morphology_summary(
                bank(d4_transform(image, *element))
            )
            torch.testing.assert_close(transformed, summary, rtol=2e-5, atol=2e-5)

        for core in ("quantum", "classical"):
            model = D4OrbitClassifier(
                core=core,
                encoder_variant="micro",
                physics_summary="moments-spectral",
                dropout=0.0,
            )
            report = model.parameter_report()
            self.assertEqual(report["total"], 122_565)
            self.assertLessEqual(report["total"], 245_221 // 2)
            self.assertEqual(report["physics_summary_dim"], 32)
            self.assertEqual(model.orbit_projection.in_features, 128)

        source = D4OrbitClassifier(
            core="classical",
            encoder_variant="micro",
            physics_summary="moments",
        )
        target = D4OrbitClassifier(
            core="quantum",
            encoder_variant="micro",
            physics_summary="moments-spectral",
        )
        prefixes = ("physics.", "encoder.", "orbit_projection.")
        state = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if key.startswith(prefixes)
        }
        original_projection = state["orbit_projection.weight"].clone()
        adapted = []
        zero_extend_input_weight(
            state,
            target.state_dict(),
            "orbit_projection.weight",
            adapted,
        )
        self.assertEqual(tuple(state["orbit_projection.weight"].shape), (8, 128))
        torch.testing.assert_close(
            state["orbit_projection.weight"][:, :112],
            original_projection,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["orbit_projection.weight"][:, 112:],
            torch.zeros(8, 16),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(adapted[0]["key"], "orbit_projection.weight")
        incompatible = target.load_state_dict(state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        with torch.no_grad():
            target.orbit_projection.weight[:, 112:].copy_(
                torch.linspace(-0.2, 0.2, 8 * 16).reshape(8, 16)
            )
        target.eval()
        reference_logits = target(image[:1])
        transformed_logits = target(d4_transform(image[:1], 1, 1))
        torch.testing.assert_close(
            transformed_logits, reference_logits, rtol=3e-5, atol=3e-5
        )

        statistical_target = D4OrbitClassifier(
            core="quantum",
            encoder_variant="micro-stat",
            physics_summary="moments",
            reuploads=3,
        )
        statistical_state = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if key.startswith(prefixes)
        }
        old_projection = statistical_state["orbit_projection.weight"].clone()
        statistical_adapted = []
        zero_extend_input_weight(
            statistical_state,
            statistical_target.state_dict(),
            "orbit_projection.weight",
            statistical_adapted,
            insert_before_tail=16,
        )
        remapped = statistical_state["orbit_projection.weight"]
        self.assertEqual(tuple(remapped.shape), (8, 208))
        torch.testing.assert_close(remapped[:, :96], old_projection[:, :96])
        torch.testing.assert_close(remapped[:, 96:192], torch.zeros(8, 96))
        torch.testing.assert_close(remapped[:, 192:], old_projection[:, 96:])
        self.assertEqual(
            statistical_adapted[0]["method"],
            "copy-prefix-and-tail-zero-inserted-inputs",
        )

    def test_annular_haar_candidate_definition_budget_and_invariance(self):
        image = torch.rand(2, 1, 96, 96)
        bank = PhysicsChannelBank()
        physics = bank(image)
        summary = annular_haar_scattering_summary(physics)
        self.assertEqual(tuple(summary.shape), (2, 104))
        self.assertTrue(torch.isfinite(summary).all())

        # Feature zero is d=1, horizontal centered difference, radius [0,4).
        log_intensity = physics[:, 1]
        padded = torch.nn.functional.pad(
            log_intensity[:, None], (1, 1, 1, 1), mode="reflect"
        )[:, 0]
        response = torch.log1p(
            8.0
            * (
                padded[:, 1:97, 2:98]
                - padded[:, 1:97, 0:96]
            ).abs()
        )
        coordinate = torch.arange(96, dtype=response.dtype) - 47.5
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        first_annulus = (xx.square() + yy.square()).sqrt() < 4.0
        expected_first = response[:, first_annulus].mean(dim=1)
        torch.testing.assert_close(summary[:, 0], expected_first)

        half_ceiling = 245_221 // 2
        model = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        report = model.parameter_report()
        self.assertEqual(report["total"], 122_595)
        self.assertEqual(report["encoder"], 118_218)
        self.assertEqual(report["orbit_projection"], 2_984)
        self.assertEqual(report["quantum"], 88)
        self.assertEqual(report["head_and_context"], 1_305)
        self.assertEqual(report["physics_summary_dim"], 180)
        self.assertEqual(report["morphology_feature_dim"], 60)
        self.assertEqual(report["morphology_context_dim"], 15)
        self.assertEqual(report["haar_summary_dim"], 104)
        self.assertFalse(report["tied_mean_dispersion"])
        self.assertEqual(report["dispersion_gate_trainable"], 0)
        self.assertFalse(report["shared_late_refinement"])
        self.assertEqual(report["shared_late_refinement_gate_trainable"], 0)
        self.assertLessEqual(report["total"], half_ceiling)
        self.assertEqual(model.orbit_projection.in_features, 372)
        self.assertEqual(model.head.projection.in_features, 63)
        self.assertEqual(model.head.projection.out_features, 18)
        self.assertEqual(
            tuple(model.morphology_context_indices.tolist()),
            HAAR_MORPHOLOGY_CONTEXT_INDICES,
        )

        small = torch.rand(1, 1, 32, 32)
        model.eval()
        logits, auxiliary = model(small, return_aux=True)
        self.assertEqual(tuple(auxiliary["haar_summary"].shape), (1, 8, 104))
        self.assertEqual(tuple(auxiliary["morphology_context"].shape), (1, 15))
        for element in D4_ELEMENTS:
            actual = model(d4_transform(small, *element))
            torch.testing.assert_close(actual, logits, rtol=5e-5, atol=5e-5)

        with self.assertRaisesRegex(
            ValueError, "requires moments-morphology-haar"
        ):
            D4OrbitClassifier(encoder_variant="deep-se-haar-morph")
        with self.assertRaisesRegex(
            ValueError, "requires deep-se-haar-morph"
        ):
            D4OrbitClassifier(physics_summary="moments-morphology-haar")

    def test_annular_haar_exact_morphology_kd_remap_and_normalization(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-morph",
            physics_summary="moments-morphology",
            dropout=0.0,
        )
        target = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            dropout=0.0,
        )
        with torch.no_grad():
            source.morphology_mean.copy_(torch.linspace(-0.5, 0.5, 60))
            source.morphology_scale.copy_(torch.linspace(0.5, 1.5, 60))
            source.head.projection.weight[:, 48:].zero_()
            source.head.classifier.weight.fill_(1.0)
            for rank, index in enumerate(HAAR_MORPHOLOGY_CONTEXT_INDICES):
                source.head.projection.weight[
                    0, 48 + index
                ] = len(HAAR_MORPHOLOGY_CONTEXT_INDICES) - rank

        source_state = source.state_dict()
        self.assertEqual(
            morphology_path_sensitivity_order(source_state),
            HAAR_MORPHOLOGY_CONTEXT_INDICES,
        )
        remapped, adapted = remap_morphology_kd_to_haar_candidate(
            source_state, target.state_dict()
        )
        self.assertEqual(
            adapted[0]["method"],
            "copy-encoder-moments-morphology-zero-new-haar",
        )
        torch.testing.assert_close(
            remapped["orbit_projection.weight"][:, :268],
            source_state["orbit_projection.weight"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            remapped["orbit_projection.weight"][:, 268:],
            torch.zeros(8, 104),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            remapped["head.projection.weight"][:, :48],
            source_state["head.projection.weight"][:, :48],
            rtol=0.0,
            atol=0.0,
        )
        selected = torch.tensor(HAAR_MORPHOLOGY_CONTEXT_INDICES)
        torch.testing.assert_close(
            remapped["head.projection.weight"][:, 48:],
            source_state["head.projection.weight"][:, 48:].index_select(
                1, selected
            ),
            rtol=0.0,
            atol=0.0,
        )
        target.load_state_dict(remapped, strict=True)
        self.assertTrue(torch.equal(target.morphology_mean, source.morphology_mean))
        self.assertTrue(torch.equal(target.morphology_scale, source.morphology_scale))

        replay = torch.rand(2, 1, 32, 32)
        source.eval()
        target.eval()
        with torch.no_grad():
            _, source_angles = source.orbit_encode(replay)
            _, target_angles = target.orbit_encode(replay)
        torch.testing.assert_close(
            target_angles, source_angles, rtol=2e-6, atol=2e-6
        )

        preserved_mean = target.morphology_mean.detach().clone()
        preserved_scale = target.morphology_scale.detach().clone()
        normalization_images = torch.rand(2, 1, 96, 96)
        loader = [
            (
                normalization_images,
                torch.tensor([0, 1]),
                torch.tensor([11, 29]),
            )
        ]
        normalization = fit_haar_morphology_normalization(
            target,
            loader,
            torch.device("cpu"),
            preserve_morphology=True,
        )
        self.assertEqual(normalization["haar"]["fit_images"], 2)
        self.assertEqual(normalization["haar"]["fit_views"], 16)
        self.assertEqual(normalization["morphology"]["fit_images"], 0)
        self.assertEqual(normalization["morphology"]["fit_views"], 0)
        self.assertTrue(normalization["morphology_preserved_from_initialization"])
        self.assertTrue(torch.equal(target.morphology_mean, preserved_mean))
        self.assertTrue(torch.equal(target.morphology_scale, preserved_scale))
        self.assertTrue(torch.isfinite(target.haar_mean).all())
        self.assertTrue(torch.isfinite(target.haar_scale).all())
        self.assertTrue((target.haar_scale > 0).all())

    def test_invariant_haar_subtype_pool_and_train_only_selection(self):
        views = torch.randn(4, 8, 104)
        reference = invariant_annular_haar_coefficients(views)
        self.assertEqual(tuple(reference.shape), (4, 56))
        for element in D4_ELEMENTS:
            permutation = right_regular_permutation(element)
            actual = invariant_annular_haar_coefficients(
                views.index_select(1, permutation)
            )
            torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)

        torch.manual_seed(20260726)
        features = torch.randn(60, 56)
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
        features[:, 1] = features[:, 0]
        features[labels == 0, :2] += 4.0
        features[labels == 1, :2] -= 4.0
        selected, center, scale, report = select_haar_subtype_coefficients(
            features, labels
        )
        self.assertEqual(tuple(selected.shape), (15,))
        self.assertEqual(tuple(center.shape), (15,))
        self.assertEqual(tuple(scale.shape), (15,))
        self.assertEqual(selected[:2].tolist(), [0, 1])
        self.assertEqual(report["class_counts"], {"axion": 20, "cdm": 20})
        self.assertEqual(report["no_sub_selection_samples"], 0)

        perturbed = features.clone()
        perturbed[labels == 2] = 1e6 * torch.randn_like(
            perturbed[labels == 2]
        )
        selected_again, center_again, scale_again, _ = (
            select_haar_subtype_coefficients(perturbed, labels)
        )
        torch.testing.assert_close(selected_again, selected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(center_again, center, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scale_again, scale, rtol=0.0, atol=0.0)

    def test_haar_subtype_residual_budget_replay_sign_gradient_and_selection(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            dropout=0.0,
        )
        target = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            haar_subtype_residual=True,
            dropout=0.0,
        )
        remapped, adapted = remap_haar_to_subtype_residual(
            source.state_dict(), target.state_dict()
        )
        self.assertEqual(
            adapted[0]["method"],
            "zero-new-invariant-haar-subtype-residual",
        )
        target.load_state_dict(remapped, strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), msg=key)
        self.assertTrue(
            torch.equal(
                target.haar_subtype_residual.weight,
                torch.zeros(15),
            )
        )

        for core in ("quantum", "classical"):
            candidate = D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_summary="moments-morphology-haar",
                core=core,
                haar_subtype_residual=True,
                dropout=0.0,
            )
            report = candidate.parameter_report()
            self.assertEqual(report["total"], 122_610)
            self.assertEqual(report["core"], 88)
            self.assertEqual(report["haar_subtype_residual_trainable"], 15)
            self.assertFalse(report["tied_mean_dispersion"])

        images = torch.rand(6, 1, 32, 32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loader = [(images, labels, torch.arange(6))]
        preserved = {
            name: getattr(target, name).detach().clone()
            for name in (
                "morphology_mean",
                "morphology_scale",
                "haar_mean",
                "haar_scale",
            )
        }
        selection = fit_haar_subtype_selection(
            target, loader, torch.device("cpu")
        )
        self.assertEqual(selection["fit_images"], 6)
        self.assertEqual(selection["fit_views"], 48)
        self.assertEqual(selection["selection_samples"], 4)
        self.assertEqual(selection["no_sub_selection_samples"], 0)
        self.assertEqual(len(selection["selected_indices"]), 15)
        self.assertEqual(len(selection["selection_spec_sha256"]), 64)
        for name, value in preserved.items():
            self.assertTrue(torch.equal(getattr(target, name), value), msg=name)

        replay_images = images[:2]
        replay = haar_subtype_exact_replay(source, target, replay_images)
        self.assertTrue(replay["exact_logits"])
        self.assertEqual(replay["max_logit_absolute_difference"], 0.0)

        source.eval()
        target.eval()
        with torch.no_grad():
            target.haar_subtype_residual.weight[0] = 0.25
            base_logits = source(replay_images)
            residual_logits, auxiliary = target(
                replay_images, return_aux=True
            )
        delta = auxiliary["haar_subtype_delta"]
        expected = torch.stack((delta, -delta, torch.zeros_like(delta)), dim=1)
        torch.testing.assert_close(
            residual_logits - base_logits, expected, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            residual_logits[:, 2], base_logits[:, 2], rtol=0.0, atol=0.0
        )

        target.train()
        target.zero_grad(set_to_none=True)
        target(replay_images).square().mean().backward()
        gradient = target.haar_subtype_residual.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)

        target.eval()
        reference_logits = target(replay_images[:1])
        for element in D4_ELEMENTS:
            transformed = target(d4_transform(replay_images[:1], *element))
            torch.testing.assert_close(
                transformed, reference_logits, rtol=5e-5, atol=5e-5
            )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_summary="moments-morphology-haar",
                tied_mean_dispersion=True,
                haar_subtype_residual=True,
            )
        tied = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            tied_mean_dispersion=True,
        )
        with self.assertRaisesRegex(ValueError, "tied dispersion"):
            remap_haar_to_subtype_residual(source.state_dict(), tied.state_dict())

    def test_shared_late_refinement_budget_replay_depth_gradients_and_symmetry(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            dropout=0.0,
        )
        target = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            shared_late_refinement=True,
            dropout=0.0,
        )
        remapped, adapted = remap_haar_to_shared_late_refinement(
            source.state_dict(), target.state_dict()
        )
        self.assertEqual(
            adapted[0]["method"],
            "zero-new-shared-late-refinement-gates",
        )
        self.assertEqual(adapted[0]["shared_block_applications"], [5, 5, 7, 7])
        target.load_state_dict(remapped, strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), msg=key)
        self.assertTrue(
            torch.equal(
                target.encoder.shared_refinement_gates,
                torch.zeros(4),
            )
        )

        report = target.parameter_report()
        self.assertEqual(report["total"], 122_599)
        self.assertEqual(report["encoder"], 118_222)
        self.assertEqual(report["orbit_projection"], 2_984)
        self.assertEqual(report["quantum"], 88)
        self.assertEqual(report["head_and_context"], 1_305)
        self.assertTrue(report["shared_late_refinement"])
        self.assertEqual(report["shared_late_refinement_gate_trainable"], 4)
        self.assertFalse(report["tied_mean_dispersion"])
        self.assertFalse(report["haar_subtype_residual"])
        self.assertLessEqual(report["total"], 245_221 // 2)
        self.assertIn(
            id(target.encoder.shared_refinement_gates),
            {id(parameter) for parameter in target.encoder.parameters()},
        )
        matched_classical = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            core="classical",
            shared_late_refinement=True,
            dropout=0.0,
        )
        self.assertEqual(matched_classical.parameter_report()["total"], 122_599)
        self.assertEqual(matched_classical.parameter_report()["core"], 88)

        initialization = shared_late_refinement_initialization_record(target)
        self.assertEqual(initialization["gate_parameters"], 4)
        self.assertTrue(initialization["all_gates_zero_after_remap"])
        self.assertEqual(
            initialization["shared_block_applications"], [5, 5, 7, 7]
        )

        images = torch.rand(2, 1, 48, 48)
        replay = shared_late_refinement_exact_replay(source, target, images)
        self.assertTrue(replay["float32"]["exact_logits"])
        self.assertTrue(replay["bfloat16_autocast"]["exact_logits"])
        self.assertEqual(
            replay["bfloat16_autocast"]["max_logit_absolute_difference"],
            0.0,
        )

        call_counts = {5: 0, 7: 0}

        def count_block(index):
            def hook(_module, _inputs, _output):
                call_counts[index] += 1
            return hook

        handles = [
            target.encoder.blocks[index].register_forward_hook(count_block(index))
            for index in (5, 7)
        ]
        try:
            with torch.no_grad():
                target.encoder(torch.rand(1, 8, 48, 48))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(call_counts, {5: 3, 7: 3})

        source.eval()
        target.eval()
        with torch.no_grad():
            base_logits, base_auxiliary = source(images, return_aux=True)
            target.encoder.shared_refinement_gates.copy_(
                torch.linspace(-0.2, 0.2, 4)
            )
            refined_logits, refined_auxiliary = target(
                images, return_aux=True
            )
        self.assertGreater(
            float(
                (
                    refined_auxiliary["encoded"]
                    - base_auxiliary["encoded"]
                ).abs().max()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                (
                    refined_auxiliary["angles"]
                    - base_auxiliary["angles"]
                ).abs().max()
            ),
            0.0,
        )
        torch.testing.assert_close(
            refined_auxiliary["morphology_context"],
            base_auxiliary["morphology_context"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            refined_auxiliary["haar_summary"],
            base_auxiliary["haar_summary"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(float((refined_logits - base_logits).abs().max()), 0.0)

        target.train()
        target.zero_grad(set_to_none=True)
        target(images).square().mean().backward()
        gate_gradient = target.encoder.shared_refinement_gates.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(gate_gradient).all())
        self.assertGreater(float(gate_gradient.norm()), 0.0)

        target.eval()
        with torch.no_grad():
            reference = target(images[:1])
            for element in D4_ELEMENTS:
                transformed = target(d4_transform(images[:1], *element))
                torch.testing.assert_close(
                    transformed, reference, rtol=5e-5, atol=5e-5
                )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_summary="moments-morphology-haar",
                shared_late_refinement=True,
                tied_mean_dispersion=True,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_summary="moments-morphology-haar",
                shared_late_refinement=True,
                haar_subtype_residual=True,
            )

    def test_annular_haar_tied_mean_dispersion_budget_replay_gradients_and_symmetry(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-morph",
            physics_summary="moments-morphology",
            dropout=0.0,
        )
        with torch.no_grad():
            source.head.projection.weight[:, 48:].zero_()
            source.head.classifier.weight.fill_(1.0)
            for rank, index in enumerate(HAAR_MORPHOLOGY_CONTEXT_INDICES):
                source.head.projection.weight[
                    0, 48 + index
                ] = len(HAAR_MORPHOLOGY_CONTEXT_INDICES) - rank

        base = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            dropout=0.0,
        )
        candidate = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            tied_mean_dispersion=True,
            dropout=0.0,
        )
        base_state, base_adapted = remap_morphology_kd_to_haar_candidate(
            source.state_dict(), base.state_dict()
        )
        candidate_state, candidate_adapted = (
            remap_morphology_kd_to_haar_candidate(
                source.state_dict(), candidate.state_dict()
            )
        )
        base.load_state_dict(base_state, strict=True)
        candidate.load_state_dict(candidate_state, strict=True)

        self.assertEqual(base.parameter_report()["total"], 122_595)
        report = candidate.parameter_report()
        self.assertEqual(report["total"], 122_603)
        self.assertEqual(report["encoder"], 118_218)
        self.assertEqual(report["orbit_projection"], 2_984)
        self.assertEqual(report["quantum"], 88)
        self.assertEqual(report["head_and_context"], 1_305)
        self.assertTrue(report["tied_mean_dispersion"])
        self.assertEqual(report["dispersion_gate_trainable"], 8)
        self.assertLessEqual(report["total"], 245_221 // 2)
        matched_classical = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            core="classical",
            tied_mean_dispersion=True,
            dropout=0.0,
        )
        self.assertEqual(matched_classical.parameter_report()["total"], 122_603)
        self.assertEqual(matched_classical.parameter_report()["core"], 88)
        self.assertNotIn(
            "zero-new-tied-mean-dispersion-gates",
            {item["method"] for item in base_adapted},
        )
        self.assertIn(
            "zero-new-tied-mean-dispersion-gates",
            {item["method"] for item in candidate_adapted},
        )
        initialization = tied_mean_dispersion_initialization_record(candidate)
        self.assertEqual(initialization["gate_parameters"], 8)
        self.assertTrue(initialization["all_gates_zero_after_remap"])
        self.assertTrue(initialization["zero_gate_exact_base_replay"])

        image = torch.rand(1, 1, 48, 48)
        base.eval()
        candidate.eval()
        with torch.no_grad():
            base_encoded, base_angles = base.orbit_encode(image)
            candidate_encoded, zero_gate_angles = candidate.orbit_encode(image)
            base_logits = base(image)
            candidate_logits = candidate(image)
        torch.testing.assert_close(
            candidate_encoded, base_encoded, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            zero_gate_angles, base_angles, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            candidate_logits, base_logits, rtol=0.0, atol=0.0
        )
        with torch.no_grad(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            _, base_autocast_angles = base.orbit_encode(image)
            _, candidate_autocast_angles = candidate.orbit_encode(image)
        torch.testing.assert_close(
            candidate_autocast_angles,
            base_autocast_angles,
            rtol=0.0,
            atol=0.0,
        )

        with torch.no_grad():
            candidate.dispersion_gates.copy_(
                torch.linspace(-0.25, 0.25, 8)
            )
            nonzero_encoded, nonzero_angles = candidate.orbit_encode(image)
        torch.testing.assert_close(
            nonzero_encoded, candidate_encoded, rtol=0.0, atol=0.0
        )
        self.assertGreater(
            float((nonzero_angles - zero_gate_angles).abs().max()), 0.0
        )

        candidate.zero_grad(set_to_none=True)
        _, train_angles = candidate.orbit_encode(image)
        train_angles.square().mean().backward()
        gate_gradient = candidate.dispersion_gates.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(gate_gradient).all())
        self.assertGreater(float(gate_gradient.norm()), 0.0)

        candidate.eval()
        with torch.no_grad():
            reference_logits, reference_auxiliary = candidate(
                image, return_aux=True
            )
            reference_encoded = reference_auxiliary["encoded"]
            reference_angles = reference_auxiliary["angles"]
            for element in D4_ELEMENTS:
                transformed = d4_transform(image, *element)
                actual_logits, actual_auxiliary = candidate(
                    transformed, return_aux=True
                )
                actual_encoded = actual_auxiliary["encoded"]
                actual_angles = actual_auxiliary["angles"]
                permutation = right_regular_permutation(element)
                torch.testing.assert_close(
                    actual_encoded,
                    reference_encoded.index_select(1, permutation),
                    rtol=2e-5,
                    atol=2e-5,
                )
                torch.testing.assert_close(
                    actual_angles,
                    reference_angles.index_select(-1, permutation),
                    rtol=2e-5,
                    atol=2e-5,
                )
                torch.testing.assert_close(
                    actual_logits, reference_logits, rtol=5e-5, atol=5e-5
                )

        with self.assertRaisesRegex(
            ValueError, "requires the deep-se-haar-morph"
        ):
            D4OrbitClassifier(tied_mean_dispersion=True)

    def test_annular_haar_matched_cores_restore_fresh_seeded_state(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-morph",
            physics_summary="moments-morphology",
            dropout=0.0,
        )
        with torch.no_grad():
            source.head.projection.weight[:, 48:].zero_()
            source.head.classifier.weight.fill_(1.0)
            for rank, index in enumerate(HAAR_MORPHOLOGY_CONTEXT_INDICES):
                source.head.projection.weight[
                    0, 48 + index
                ] = len(HAAR_MORPHOLOGY_CONTEXT_INDICES) - rank
        source_state = source.state_dict()

        targets = {}
        fresh_states = {}
        noncore_before_restore = {}
        reports = {}
        for core in ("quantum", "classical"):
            torch.manual_seed(20260725)
            target = D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_summary="moments-morphology-haar",
                core=core,
                dropout=0.0,
            )
            fresh = clone_module_state(target.core)
            remapped, _ = remap_morphology_kd_to_haar_candidate(
                source_state, target.state_dict()
            )
            target.load_state_dict(remapped, strict=True)
            noncore_before_restore[core] = {
                key: value.detach().clone()
                for key, value in target.state_dict().items()
                if not key.startswith("core.")
            }
            reports[core] = restore_fresh_haar_core(target, fresh)
            targets[core] = target
            fresh_states[core] = fresh

        self.assertFalse(
            torch.equal(
                fresh_states["quantum"]["params"],
                fresh_states["classical"]["params"],
            )
        )
        for core, target in targets.items():
            self.assertEqual(target.parameter_report()["total"], 122_595)
            self.assertEqual(target.parameter_report()["core"], 88)
            self.assertEqual(reports[core]["trainable_parameters"], 88)
            self.assertTrue(reports[core]["restored_exactly"])
            for key, expected in fresh_states[core].items():
                self.assertTrue(
                    torch.equal(target.core.state_dict()[key], expected),
                    msg=f"{core} core tensor was not restored: {key}",
                )
            for key, expected in noncore_before_restore[core].items():
                self.assertTrue(
                    torch.equal(target.state_dict()[key], expected),
                    msg=f"{core} non-core tensor changed during restore: {key}",
                )

        quantum_noncore = {
            key: value
            for key, value in targets["quantum"].state_dict().items()
            if not key.startswith("core.")
        }
        classical_noncore = {
            key: value
            for key, value in targets["classical"].state_dict().items()
            if not key.startswith("core.")
        }
        self.assertEqual(set(quantum_noncore), set(classical_noncore))
        for key in quantum_noncore:
            self.assertTrue(
                torch.equal(quantum_noncore[key], classical_noncore[key]),
                msg=f"Matched controls differ outside the core: {key}",
            )

        image = torch.rand(1, 1, 32, 32)
        for core, target in targets.items():
            target.train()
            target.zero_grad(set_to_none=True)
            target(image).square().mean().backward()
            gradient = target.core.params.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.0, msg=core)

    def test_lens_morphology_summary_is_finite_and_d4_invariant(self):
        image = torch.rand(3, 1, 96, 96)
        bank = PhysicsChannelBank()
        summary = lens_morphology_summary(bank(image))
        self.assertEqual(tuple(summary.shape), (3, 60))
        self.assertTrue(torch.isfinite(summary).all())
        for element in D4_ELEMENTS:
            transformed = lens_morphology_summary(
                bank(d4_transform(image, *element))
            )
            torch.testing.assert_close(
                transformed, summary, rtol=2e-5, atol=2e-5
            )

    def test_spectral_morphology_summary_has_no_empty_ring_artifact(self):
        bank = PhysicsChannelBank()
        blank = torch.zeros(1, 1, 96, 96)
        impulse = blank.clone()
        impulse[:, :, 48, 48] = 1.0
        for image in (blank, impulse):
            summary = spectral_morphology_summary(bank(image))
            self.assertTrue(torch.isfinite(summary).all())
            self.assertTrue((summary[:, 12:] >= 0.0).all())
            self.assertTrue((summary[:, 12:] <= 1.0 + 1e-6).all())
        self.assertLess(
            float(spectral_morphology_summary(bank(blank))[:, 12:].abs().max()),
            1e-6,
        )

    def test_hash_ranked_reduced_data_subsets_are_nested(self):
        labels = np.repeat(np.arange(3), 10)
        indices = np.arange(len(labels), dtype=np.int64)
        manifest_sha = "a" * 64
        small = hash_ranked_subset(indices, labels, 3, manifest_sha)
        large = hash_ranked_subset(indices, labels, 6, manifest_sha)
        self.assertEqual(len(small), 9)
        self.assertEqual(len(large), 18)
        self.assertTrue(set(small.tolist()).issubset(large.tolist()))
        self.assertEqual(np.bincount(labels[small], minlength=3).tolist(), [3, 3, 3])
        self.assertEqual(
            index_membership_sha256(small), index_membership_sha256(small[::-1])
        )

    def test_stratified_oof_folds_are_stable_balanced_and_exhaustive(self):
        labels = np.repeat(np.arange(3), 11)
        indices = np.arange(len(labels), dtype=np.int64)
        folds = stratified_hash_folds(indices, labels, 2, "b" * 64)
        repeated = stratified_hash_folds(indices[::-1], labels, 2, "b" * 64)
        self.assertEqual(len(folds), 2)
        for actual, expected in zip(folds, repeated):
            self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(
            np.array_equal(np.sort(np.concatenate(folds)), indices)
        )
        self.assertEqual(np.intersect1d(folds[0], folds[1]).size, 0)
        self.assertEqual(
            np.bincount(labels[folds[0]], minlength=3).tolist(), [6, 5, 6]
        )
        self.assertEqual(
            np.bincount(labels[folds[1]], minlength=3).tolist(), [5, 6, 5]
        )

    def test_radial_physics_variant_is_compact_and_model_invariant(self):
        image = torch.rand(2, 1, 32, 32)
        bank = PhysicsChannelBank(variant="radial")
        reference = bank(image)
        self.assertEqual(reference.shape[1], 10)
        model = D4OrbitClassifier(physics_variant="radial")
        self.assertLess(model.parameter_report()["total"], 250_000)
        model.eval()
        reference_logits = model(image)
        for element in D4_ELEMENTS:
            actual_logits = model(d4_transform(image, *element))
            torch.testing.assert_close(
                actual_logits, reference_logits, rtol=2e-5, atol=2e-5
            )

    def test_vectorized_lenspinn_preprocessing_is_finite(self):
        image = torch.rand(3, 1, 32, 32)
        image[:, :, :4, :4] = 0
        distortion = lenspinn_distortion(image)
        self.assertTrue(torch.isfinite(distortion).all())
        tokenizer = ShiftPatchTokenizer(32, 32, 16)
        patches = tokenizer(image)
        inversion = ScalarLensInversion(32, 32, tokenizer.num_patches, heads=4)
        radius, source = inversion(image, patches)
        self.assertEqual(tuple(radius.shape), (3, 1))
        self.assertEqual(tuple(source.shape), (3, 1, 32, 32))
        self.assertTrue(torch.isfinite(source).all())

    def test_repaired_lenspinn_splat_is_centered_and_differentiable(self):
        image = torch.rand(2, 1, 32, 32)
        tokenizer = ShiftPatchTokenizer(32, 32, 16)
        inversion = ScalarLensInversion(
            32,
            32,
            tokenizer.num_patches,
            heads=4,
            reconstruction="differentiable",
        )
        identity = inversion.image_to_source(image, torch.zeros(2, 1))
        torch.testing.assert_close(identity, image, rtol=2e-5, atol=2e-5)

        _, source = inversion(image, tokenizer(image))
        weights = torch.linspace(0.0, 1.0, 32).view(1, 1, 1, 32)
        (source * weights).mean().backward()
        tokenizer_grad = sum(
            float(parameter.grad.square().sum())
            for parameter in tokenizer.parameters()
            if parameter.grad is not None
        )
        inversion_grad = sum(
            float(parameter.grad.square().sum())
            for parameter in inversion.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(tokenizer_grad, 0.0)
        self.assertGreater(inversion_grad, 0.0)

    def test_paired_statistics_detect_better_predictions(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        perfect = np.eye(3)[labels] * 5.0
        wrong = np.roll(perfect, 1, axis=1)
        bootstrap = paired_bootstrap_accuracy(labels, perfect, wrong, samples=200, seed=1)
        self.assertEqual(bootstrap["difference"], 1.0)
        self.assertEqual(bootstrap["ci95_low"], 1.0)
        test = mcnemar_exact(labels, perfect, wrong)
        self.assertEqual(test["a_correct_b_wrong"], 6)
        self.assertEqual(test["a_wrong_b_correct"], 0)

    def test_probability_ensemble_uses_probability_space(self):
        first = np.array([[8.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        second = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        combined = probability_ensemble_logits((first, second))
        actual = np.exp(combined)
        expected = np.mean(
            [
                np.exp(value - value.max(1, keepdims=True))
                / np.exp(value - value.max(1, keepdims=True)).sum(1, keepdims=True)
                for value in (first, second)
            ],
            axis=0,
        )
        np.testing.assert_allclose(actual, expected)

    def test_hierarchical_loss_is_finite_and_trains_all_classes(self):
        logits = torch.randn(9, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        loss = hierarchical_model_i_loss(logits, targets, label_smoothing=0.02)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0.0)

    def test_subtype_specialist_filters_after_parent_subset_is_fixed(self):
        labels = np.array([0, 1, 2, 1, 0, 2, 0, 2, 1], dtype=np.int64)
        train = np.array([8, 2, 4, 5, 1, 0], dtype=np.int64)
        validation = np.array([7, 6, 3], dtype=np.int64)
        parent_train_hash = index_membership_sha256(train)
        subtype_train, subtype_validation, names, report = (
            select_model_i_subtype_task(
                train, validation, labels, ["axion", "cdm", "no_sub"]
            )
        )
        np.testing.assert_array_equal(subtype_train, np.array([8, 4, 1, 0]))
        np.testing.assert_array_equal(subtype_validation, np.array([6, 3]))
        self.assertEqual(names, ["axion", "cdm"])
        self.assertEqual(report["parent_train_membership_sha256"], parent_train_hash)
        self.assertEqual(
            report["subtype_train_membership_sha256"],
            index_membership_sha256(subtype_train),
        )
        self.assertEqual(report["parent_train_size"], 6)
        self.assertEqual(report["task"], "axion-vs-cdm-specialist")
        with self.assertRaisesRegex(ValueError, "class order"):
            select_model_i_subtype_task(
                train, validation, labels, ["cdm", "axion", "no_sub"]
            )

    def test_knowledge_distillation_loss_is_scaled_and_teacher_is_frozen(self):
        student = torch.randn(7, 3, requires_grad=True)
        teacher = torch.randn(7, 3, requires_grad=True)
        temperature = 2.0
        actual = knowledge_distillation_loss(student, teacher, temperature)
        expected = temperature**2 * torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student.float() / temperature, dim=1),
            torch.nn.functional.softmax(teacher.detach().float() / temperature, dim=1),
            reduction="batchmean",
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.norm()), 0.0)
        self.assertIsNone(teacher.grad)
        identical = knowledge_distillation_loss(teacher, teacher, temperature)
        self.assertLess(abs(float(identical)), 1e-6)

        ensemble_student = torch.randn(5, 3, requires_grad=True)
        teacher_a = torch.randn(5, 3, requires_grad=True)
        teacher_b = torch.randn(5, 3, requires_grad=True)
        ensemble_loss = knowledge_distillation_loss(
            ensemble_student, (teacher_a, teacher_b), temperature
        )
        ensemble_target = torch.stack(
            (
                torch.softmax(teacher_a.detach() / temperature, dim=1),
                torch.softmax(teacher_b.detach() / temperature, dim=1),
            )
        ).mean(dim=0)
        ensemble_expected = temperature**2 * torch.nn.functional.kl_div(
            torch.log_softmax(ensemble_student / temperature, dim=1),
            ensemble_target,
            reduction="batchmean",
        )
        torch.testing.assert_close(ensemble_loss, ensemble_expected)
        ensemble_loss.backward()
        self.assertIsNotNone(ensemble_student.grad)
        self.assertIsNone(teacher_a.grad)
        self.assertIsNone(teacher_b.grad)

        teacher_model = D4OrbitClassifier(
            core="classical",
            encoder_variant="tiny",
            physics_summary="moments",
            include_context=True,
        )
        student_model = D4OrbitClassifier(
            core="quantum",
            encoder_variant="micro",
            physics_summary="moments",
        )
        self.assertEqual(teacher_model.parameter_report()["total"], 272_933)
        self.assertEqual(student_model.parameter_report()["total"], 122_437)
        self.assertFalse(
            any(key.startswith("teacher") for key in student_model.state_dict())
        )

        oof_student = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])
        morphology = torch.tensor(
            [[4.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 4.0], [3.0, 1.0, 0.0]],
            requires_grad=True,
        )
        spatial = torch.tensor(
            [[0.0, 4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
            requires_grad=True,
        )
        oof_per_sample, valid, oof_targets = (
            correctness_gated_oof_distillation_loss(
                oof_student, morphology, spatial, targets, 2.0
            )
        )
        self.assertEqual(valid.tolist(), [True, True, True, False])
        self.assertEqual(float(oof_per_sample[-1]), 0.0)
        torch.testing.assert_close(
            oof_targets[valid].sum(1), torch.ones(3), rtol=1e-6, atol=1e-6
        )
        oof_per_sample.mean().backward()
        self.assertGreater(float(oof_student.grad.norm()), 0.0)
        self.assertIsNone(morphology.grad)
        self.assertIsNone(spatial.grad)

    def test_physics_augmentation_is_finite_shape_preserving_and_optional(self):
        images = torch.rand(6, 1, 24, 24)
        unchanged = physics_augment_batch(images)
        torch.testing.assert_close(unchanged, images, rtol=0.0, atol=0.0)
        augmented = physics_augment_batch(
            images,
            photon_probability=1.0,
            photon_count_min=128.0,
            photon_count_max=128.0,
            psf_probability=1.0,
            read_noise_std=0.001,
        )
        self.assertEqual(tuple(augmented.shape), tuple(images.shape))
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue((augmented >= 0).all())
        self.assertGreater(float((augmented - images).abs().mean()), 0.0)

    def test_subtype_mixup_preserves_no_substructure_and_soft_targets(self):
        images = torch.arange(6.0).view(6, 1, 1, 1).expand(-1, 1, 4, 4)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        torch.manual_seed(19)
        mixed, probabilities, count, mean_anchor = subtype_mixup_batch(
            images,
            targets,
            probability=1.0,
            alpha=0.4,
        )
        self.assertEqual(count, 4)
        self.assertGreaterEqual(mean_anchor, 0.5)
        self.assertLessEqual(mean_anchor, 1.0)
        torch.testing.assert_close(
            mixed[targets == 2], images[targets == 2], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones(6), rtol=1e-6, atol=1e-6
        )
        self.assertTrue((probabilities[targets < 2, :2] > 0).all())
        self.assertTrue((probabilities[targets < 2, 2] == 0).all())
        torch.testing.assert_close(
            probabilities[targets == 2, 2],
            torch.ones(2),
            rtol=0.0,
            atol=0.0,
        )

        logits = torch.randn(6, 3, requires_grad=True)
        loss = soft_target_cross_entropy(
            logits, probabilities, label_smoothing=0.02
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad.norm()), 0.0)

        unchanged, one_hot, unchanged_count, unchanged_weight = subtype_mixup_batch(
            images, targets, probability=0.0, alpha=0.4
        )
        torch.testing.assert_close(unchanged, images)
        torch.testing.assert_close(
            one_hot, torch.nn.functional.one_hot(targets, 3).float()
        )
        self.assertEqual(unchanged_count, 0)
        self.assertEqual(unchanged_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
