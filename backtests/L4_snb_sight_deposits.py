"""
L4 SNB weekly sight deposits — cube ID not discoverable via free API probe.

The data.snb.ch portal advertises CSV/JSON downloads of every cube, but the
public catalog (https://data.snb.ch/en) is a single-page JS application —
the cube list itself is loaded via authenticated XHR and not exposed at any
documented public REST path. Probing common candidate cube IDs for the
weekly "Sichtguthaben bei der SNB" series (snbsigtl, snbsdepo, snbliqto,
snbgesb, snbres, snbsight, snbliqlo, snbliqi, ...) all return 404
{"message":"Table not found"}.

FRED has only a monthly Swiss M1/M2 (MABMM301CHM189S), not the weekly
SNB sight-deposit series since 2009.

Without the weekly cube ID we cannot reproduce the rule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "L4_snb_sight_deposits",
        ("data.snb.ch cube ID for weekly sight deposits is not discoverable "
         "via the public CSV endpoint (every probed cube returns 404). FRED "
         "has only monthly Swiss M1, which is too coarse for a 2σ weekly-"
         "change trigger."),
        extra={
            "source_attempted": "https://data.snb.ch/api/cube/<cube>/data/csv/en (probed snbsigtl/snbsdepo/snbliqto/snbres/snbsight/snbliqlo/etc.)",
            "rule": ("Weekly SNB sight-deposit change > 2σ above 52w mean -> short EUR/CHF "
                     "(long FXF / short FXE) -- intended rule, not testable here."),
            "mechanism": "SNB intervention to weaken CHF leaves a footprint as weekly sight-deposit growth; subsequent EUR/CHF appreciation tends to fade.",
        },
    )


if __name__ == "__main__":
    main()
