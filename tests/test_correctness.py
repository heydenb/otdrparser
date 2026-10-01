#!/usr/bin/env python3

import io
import pathlib
import sys
import unittest

sys.path.append("..")
import otdrparser


TEST_DIRECTORY = pathlib.Path(__file__).parent


class FixedParameterTests(unittest.TestCase):
    def parse_fixture(self, name):
        with (TEST_DIRECTORY / name).open("rb") as fp:
            return otdrparser.parse2(fp)

    def test_fixed_parameters_use_documented_units_and_signs(self):
        data = self.parse_fixture("example.sor")
        fixed = data["FxdParams"]

        self.assertEqual(fixed["averaging_time"], 1.5)
        self.assertAlmostEqual(fixed["range"], 1.4664)
        self.assertEqual(fixed["reflection_threshold"], -65.535)
        self.assertEqual(fixed["end_of_transmission_threshold"], 5.0)

    def test_distances_are_expressed_in_meters(self):
        data = self.parse_fixture("issue9.sor")

        self.assertAlmostEqual(
            data["DataPts"]["data_points"][-1][0],
            2501.387576623927,
        )
        self.assertAlmostEqual(data["FxdParams"]["range"], 1.4664)
        self.assertAlmostEqual(
            data["KeyEvents"]["events"][-1]["distance_of_travel"],
            1249.193115344144,
        )

    def test_key_event_marker_positions_are_signed(self):
        data = self.parse_fixture("example.sor")

        self.assertEqual(
            data["KeyEvents"]["events"][0]["end_of_previous_event"],
            -48875,
        )


class CompactParseTests(unittest.TestCase):
    def parse_fixture(self, parser, name="example.sor"):
        with (TEST_DIRECTORY / name).open("rb") as fp:
            return parser(fp)

    def test_compact_parse_preserves_blocks_and_metadata(self):
        expanded = self.parse_fixture(otdrparser.parse2)
        compact = self.parse_fixture(otdrparser.parse_compact)

        self.assertEqual(compact.keys(), expanded.keys())
        self.assertEqual(compact["GenParams"], expanded["GenParams"])
        self.assertEqual(compact["FxdParams"], expanded["FxdParams"])

        expanded_data = expanded["DataPts"]
        compact_data = compact["DataPts"]
        for key in (
            "name",
            "number_of_data_points",
            "number_of_traces",
            "number_of_data_points2",
            "scaling_factor",
        ):
            self.assertEqual(compact_data[key], expanded_data[key])

    def test_compact_parse_returns_one_signal_array(self):
        compact = self.parse_fixture(otdrparser.parse_compact)
        data = compact["DataPts"]

        self.assertNotIn("data_points", data)
        self.assertIn("data_point_values", data)
        self.assertEqual(
            len(data["data_point_values"]),
            data["number_of_data_points"],
        )
        self.assertGreater(data["sample_spacing_m"], 0)
        self.assertEqual(data["start_distance_m"], 0.0)

    def test_compact_signal_and_reconstructed_distances_match_parse2(self):
        expanded = self.parse_fixture(otdrparser.parse2, "issue9.sor")["DataPts"]
        compact = self.parse_fixture(
            otdrparser.parse_compact, "issue9.sor"
        )["DataPts"]

        self.assertEqual(
            compact["data_point_values"],
            [signal for _, signal in expanded["data_points"]],
        )

        for index in (0, 1, len(compact["data_point_values"]) - 1):
            reconstructed_distance = (
                compact["start_distance_m"]
                + index * compact["sample_spacing_m"]
            )
            self.assertAlmostEqual(
                reconstructed_distance,
                expanded["data_points"][index][0],
            )

    def test_existing_parsers_keep_expanded_data_points(self):
        parsed_list = self.parse_fixture(otdrparser.parse)
        parsed_dict = self.parse_fixture(otdrparser.parse2)
        list_data = next(
            block for block in parsed_list if block["name"] == "DataPts"
        )

        for data in (list_data, parsed_dict["DataPts"]):
            self.assertIn("data_points", data)
            self.assertNotIn("data_point_values", data)
            self.assertNotIn("sample_spacing_m", data)

    def test_compact_parse_accepts_in_memory_bytes(self):
        content = (TEST_DIRECTORY / "example.sor").read_bytes()

        data = otdrparser.parse_compact(io.BytesIO(content))

        self.assertIn("data_point_values", data["DataPts"])


class InvalidInputTests(unittest.TestCase):
    def test_latin_1_metadata_is_decoded(self):
        self.assertEqual(
            otdrparser.read_zero_terminated_string(io.BytesIO(b"caf\xe9\x00")),
            "café",
        )

    def test_empty_input_raises_instead_of_looping_forever(self):
        with self.assertRaisesRegex(
            otdrparser.OTDRParserError,
            "Unexpected end of file while reading a string",
        ):
            otdrparser.parse2(io.BytesIO(b""))

    def test_truncated_input_raises_descriptive_error(self):
        with (TEST_DIRECTORY / "example.sor").open("rb") as fp:
            truncated = fp.read(100)

        with self.assertRaises(otdrparser.OTDRParserError):
            otdrparser.parse2(io.BytesIO(truncated))

    def test_non_map_header_is_rejected(self):
        data = b"NotMap\x00" + b"\x00" * 20

        with self.assertRaisesRegex(
            otdrparser.OTDRParserError,
            "Unsupported file header",
        ):
            otdrparser.parse2(io.BytesIO(data))

    def test_unsupported_major_version_is_rejected(self):
        data = b"Map\x00" + (100).to_bytes(2, "little") + b"\x00" * 6

        with self.assertRaisesRegex(
            otdrparser.OTDRParserError,
            "Unsupported SOR version 1.0",
        ):
            otdrparser.parse2(io.BytesIO(data))

    def test_multiple_traces_are_rejected_instead_of_misparsed(self):
        data = b"DataPts\x00" + b"\x00" * 4 + (2).to_bytes(2, "little")

        with self.assertRaisesRegex(
            otdrparser.OTDRParserError,
            "Multiple traces are not supported",
        ):
            otdrparser.parse_datapts_block(
                io.BytesIO(data), len(data), sample_spacing=1, index_of_refraction=1
            )


class GeneralParameterTests(unittest.TestCase):
    @staticmethod
    def block_without_comments():
        return b"".join(
            (
                b"GenParams\x00",
                b"EN",
                b"cable\x00",
                b"fiber\x00",
                (657).to_bytes(2, "little", signed=True),
                (1310).to_bytes(2, "little"),
                b"location-a\x00",
                b"location-b\x00",
                b"cable-code\x00",
                b"BC",
                (0).to_bytes(4, "little", signed=True),
                (0).to_bytes(4, "little", signed=True),
                b"operator\x00",
            )
        )

    def test_genparams_accepts_completely_omitted_comments(self):
        block = self.block_without_comments()

        parsed = otdrparser.parse_genparams_block(io.BytesIO(block), len(block))

        self.assertEqual(parsed["fiber_type"], 657)
        self.assertEqual(parsed["wavelength"], 1310)
        self.assertEqual(parsed["operator"], "operator")
        self.assertEqual(parsed["comments"], "")

    def test_genparams_rejects_unterminated_comments(self):
        block = self.block_without_comments() + b"unterminated"

        with self.assertRaisesRegex(
            otdrparser.OTDRParserError,
            "Unexpected end of file while reading a string",
        ):
            otdrparser.parse_genparams_block(io.BytesIO(block), len(block))


if __name__ == "__main__":
    unittest.main()
