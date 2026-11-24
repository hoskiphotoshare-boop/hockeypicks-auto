import unittest

from src.team_geo import haversine_miles, team_coords, travel_miles, tz_offset_hours, tz_shift_hours


class TestTeamGeo(unittest.TestCase):
    def test_haversine_edm_cgy(self):
        d = haversine_miles(53.5461, -113.4938, 51.0447, -114.0547)
        self.assertAlmostEqual(d, 170.0, delta=10.0)

    def test_team_coords_unknown(self):
        self.assertIsNone(team_coords("ZZZ"))
        self.assertEqual(team_coords("EDM"), (53.5461, -113.4938))

    def test_travel_miles_unknown_neutral(self):
        self.assertEqual(travel_miles("ZZZ", "TOR"), 0.0)
        self.assertEqual(travel_miles("EDM", "ZZZ"), 0.0)

    def test_tz_offsets_summer_winter(self):
        self.assertEqual(tz_offset_hours("TOR", "2026-01-15"), -5)
        self.assertEqual(tz_offset_hours("TOR", "2026-07-15"), -4)
        self.assertEqual(tz_offset_hours("NYR", "2026-01-15"), -5)
        self.assertEqual(tz_offset_hours("NYR", "2026-07-15"), -4)
        self.assertEqual(tz_offset_hours("VAN", "2026-01-15"), -8)
        self.assertEqual(tz_offset_hours("VAN", "2026-07-15"), -7)

    def test_tz_offset_unknown_neutral(self):
        self.assertEqual(tz_offset_hours("ZZZ", "2026-01-15"), 0)

    def test_tz_shift_hours(self):
        self.assertEqual(tz_shift_hours("VAN", "TOR", "2026-01-15"), 3)

    def test_tz_offset_iana_string(self):
        self.assertEqual(tz_offset_hours("America/Toronto", "2026-01-15"), -5)


if __name__ == "__main__":
    unittest.main()
