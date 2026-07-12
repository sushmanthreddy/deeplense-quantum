import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from d4_orqb.model import D4OrbitClassifier
from d4_orqb.spatial_paired_init import (
    PROTOCOL,
    build_paired_initializers,
    construct_native_spatial_model,
    file_sha256,
    parse_args,
    state_sha256,
)


def component_state(state, core):
    return {
        name: value.detach().cpu()
        for name, value in state.items()
        if name.startswith("core.") is core
    }


def native_core_state(model):
    return {
        f"core.{name}": value.detach().cpu()
        for name, value in model.core.state_dict().items()
    }


class SpatialPairedInitializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="spatial-paired-init-")
        cls.root = Path(cls.temporary.name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(812)
            source = D4OrbitClassifier(
                num_classes=3,
                heads=4,
                reuploads=3,
                core="quantum",
                include_context=True,
                encoder_variant="micro",
                physics_variant="base",
                physics_summary="moments",
                quantum_encoding="angle",
                observable_readout="pair",
                dropout=0.10,
            ).cpu()
        cls.source_state = {
            name: value.detach().cpu().clone()
            for name, value in source.state_dict().items()
        }
        cls.backbone = cls.root / "synthetic-backbone.pt"
        torch.save({"model": cls.source_state, "epoch": 36}, cls.backbone)
        cls.backbone_sha256 = file_sha256(cls.backbone)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_builds_exact_paired_epoch_zero_checkpoints(self):
        output = self.root / "paired-seed-3"
        report = build_paired_initializers(
            self.backbone,
            output,
            [3],
            expected_backbone_sha256=self.backbone_sha256,
        )
        quantum_path = output / "seed-3" / "quantum-init.pt"
        classical_path = output / "seed-3" / "classical-init.pt"
        quantum = torch.load(quantum_path, map_location="cpu", weights_only=False)
        classical = torch.load(
            classical_path, map_location="cpu", weights_only=False
        )

        self.assertEqual(quantum["protocol_id"], PROTOCOL)
        self.assertEqual(classical["protocol_id"], PROTOCOL)
        self.assertEqual(quantum["epoch"], 0)
        self.assertEqual(classical["epoch"], 0)
        self.assertEqual(quantum["seed"], 3)
        self.assertEqual(classical["seed"], 3)
        self.assertEqual(quantum["common_head_seed"], 10003)
        self.assertEqual(classical["common_head_seed"], 10003)

        quantum_noncore = component_state(quantum["model"], core=False)
        classical_noncore = component_state(classical["model"], core=False)
        self.assertEqual(set(quantum_noncore), set(classical_noncore))
        for name in quantum_noncore:
            self.assertTrue(
                torch.equal(quantum_noncore[name], classical_noncore[name]), name
            )
        common_digest = state_sha256(quantum_noncore)
        self.assertEqual(quantum["common_noncore_state_sha256"], common_digest)
        self.assertEqual(classical["common_noncore_state_sha256"], common_digest)

        expected_quantum = native_core_state(
            construct_native_spatial_model("quantum", 3)
        )
        expected_classical = native_core_state(
            construct_native_spatial_model("classical", 3)
        )
        actual_quantum = component_state(quantum["model"], core=True)
        actual_classical = component_state(classical["model"], core=True)
        self.assertEqual(set(actual_quantum), set(expected_quantum))
        self.assertEqual(set(actual_classical), set(expected_classical))
        for name in expected_quantum:
            self.assertTrue(torch.equal(actual_quantum[name], expected_quantum[name]))
        for name in expected_classical:
            self.assertTrue(
                torch.equal(actual_classical[name], expected_classical[name])
            )
        self.assertEqual(
            quantum["native_core_state_sha256"], state_sha256(expected_quantum)
        )
        self.assertEqual(
            classical["native_core_state_sha256"], state_sha256(expected_classical)
        )

        source_projection = self.source_state["orbit_projection.weight"]
        target_projection = quantum["model"]["orbit_projection.weight"]
        self.assertEqual(tuple(source_projection.shape), (8, 112))
        self.assertEqual(tuple(target_projection.shape), (8, 208))
        torch.testing.assert_close(
            target_projection[:, :96], source_projection[:, :96], rtol=0, atol=0
        )
        torch.testing.assert_close(
            target_projection[:, 96:192],
            torch.zeros_like(target_projection[:, 96:192]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            target_projection[:, 192:], source_projection[:, 96:], rtol=0, atol=0
        )

        persisted = json.loads((output / "report.json").read_text())
        self.assertEqual(persisted, report)
        self.assertFalse(report["official_test_opened"])
        self.assertFalse(report["official_test_reference_accepted"])
        self.assertEqual(
            report["backbone_fingerprint"]["sha256"], self.backbone_sha256
        )
        self.assertEqual(
            report["per_seed"]["3"]["common_noncore"]["sha256"],
            common_digest,
        )
        seed_report = report["per_seed"]["3"]
        self.assertEqual(seed_report["schema_version"], 1)
        self.assertEqual(seed_report["protocol_id"], PROTOCOL)
        self.assertEqual(seed_report["seed"], 3)
        self.assertEqual(seed_report["architecture"], report["architecture"])
        self.assertEqual(
            seed_report["backbone_fingerprint"]["sha256"],
            self.backbone_sha256,
        )
        self.assertEqual(seed_report["common_noncore_state_sha256"], common_digest)
        self.assertEqual(seed_report["arms"]["quantum"]["core_name"], "quantum")
        self.assertEqual(
            seed_report["arms"]["classical"]["core_name"], "classical"
        )
        self.assertNotEqual(
            seed_report["arms"]["quantum"]["checkpoint_sha256"],
            seed_report["arms"]["classical"]["checkpoint_sha256"],
        )
        for arm, checkpoint in (("quantum", quantum), ("classical", classical)):
            arm_report = seed_report["arms"][arm]
            self.assertEqual(arm_report["core_name"], checkpoint["core_name"])
            self.assertEqual(
                arm_report["full_state_sha256"], checkpoint["full_state_sha256"]
            )
            self.assertEqual(
                arm_report["core_state_sha256"], checkpoint["core_state_sha256"]
            )
            self.assertEqual(
                arm_report["native_core_state_sha256"],
                checkpoint["native_core_state_sha256"],
            )
            self.assertEqual(
                arm_report["noncore_state_sha256"],
                checkpoint["noncore_state_sha256"],
            )
        self.assertEqual(
            report["per_seed"]["3"]["arms"]["quantum"]["parameters"]["total"],
            122573,
        )
        self.assertEqual(
            report["per_seed"]["3"]["arms"]["quantum"]["parameters"]["quantum"],
            132,
        )
        self.assertEqual(
            report["per_seed"]["3"]["arms"]["classical"]["parameters"]["core"],
            132,
        )
        self.assertEqual(
            report["per_seed"]["3"]["arms"]["quantum"]["checkpoint_sha256"],
            file_sha256(quantum_path),
        )
        self.assertEqual(
            report["per_seed"]["3"]["arms"]["classical"]["checkpoint_sha256"],
            file_sha256(classical_path),
        )
        payload = dict(report)
        payload_digest = payload.pop("report_payload_sha256")
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        self.assertEqual(payload_digest, hashlib.sha256(encoded).hexdigest())

    def test_refuses_an_existing_output(self):
        output = self.root / "already-exists"
        output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            build_paired_initializers(
                self.backbone,
                output,
                [0],
                expected_backbone_sha256=self.backbone_sha256,
            )

    def test_rejects_official_test_path_at_argument_parsing(self):
        forbidden_output = self.root / ("Model_I" + "_test") / "initializers"
        with self.assertRaisesRegex(ValueError, "Official-test references"):
            parse_args(
                [
                    "--backbone-checkpoint",
                    str(self.backbone),
                    "--output-dir",
                    str(forbidden_output),
                    "--seeds",
                    "0",
                    "1",
                    "2",
                ]
            )
        self.assertFalse(forbidden_output.exists())

    def test_rejects_official_test_string_embedded_in_checkpoint(self):
        unsafe = self.root / "unsafe-backbone.pt"
        torch.save(
            {
                "model": self.source_state,
                "epoch": 36,
                "provenance": {"root": "/datasets/Model_I" + "_test"},
            },
            unsafe,
        )
        output = self.root / "unsafe-output"
        with self.assertRaisesRegex(ValueError, "contains an official-test reference"):
            build_paired_initializers(
                unsafe,
                output,
                [0],
                expected_backbone_sha256=file_sha256(unsafe),
            )
        self.assertFalse(output.exists())

    def test_rejects_duplicate_seeds_without_creating_output(self):
        output = self.root / "duplicate-seeds"
        with self.assertRaisesRegex(ValueError, "Seeds must be unique"):
            build_paired_initializers(
                self.backbone,
                output,
                [1, 1],
                expected_backbone_sha256=self.backbone_sha256,
            )
        self.assertFalse(output.exists())

    def test_state_digest_is_independent_of_channels_last_stride(self):
        values = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        channels_last = values.clone().contiguous(memory_format=torch.channels_last)
        contiguous = values.clone().contiguous()
        self.assertNotEqual(channels_last.stride(), contiguous.stride())
        self.assertEqual(
            state_sha256({"weight": channels_last}),
            state_sha256({"weight": contiguous}),
        )


if __name__ == "__main__":
    unittest.main()
