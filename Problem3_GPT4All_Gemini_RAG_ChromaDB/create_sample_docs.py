"""Generate 3 sample text documents used to demonstrate the RAG pipeline.

Run once before `ingest.py`:

    python create_sample_docs.py

This creates `documents/renewable_energy.txt`, `documents/space_exploration.txt`
and `documents/personal_finance.txt`. Feel free to replace these with your own
`.txt` or `.pdf` files -- the ingestion script picks up every `*.txt` and
`*.pdf` file it finds in the `documents/` folder.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "documents"

DOCUMENTS: dict[str, str] = {
    "renewable_energy.txt": """
        Renewable Energy Basics

        Solar Power: Photovoltaic (PV) panels convert sunlight directly into
        electricity using semiconductor cells. Utility-scale solar farms now
        produce electricity at a lower levelized cost than most fossil-fuel
        plants in sunny regions. Residential rooftop systems typically pay
        back their installation cost within 6 to 10 years depending on local
        electricity rates and incentives.

        Wind Power: Modern wind turbines convert kinetic energy from moving
        air into electricity via a rotor, gearbox, and generator. Offshore
        wind farms benefit from stronger, steadier winds than onshore sites
        but cost more to install and maintain due to marine foundations and
        subsea cabling.

        Energy Storage: Because solar and wind output varies with weather
        and time of day, grid-scale battery storage (mostly lithium-ion) is
        increasingly paired with renewables to shift generation to periods
        of high demand. Pumped-hydro storage remains the largest form of
        grid storage worldwide by capacity.

        Grid Integration: Utilities use demand forecasting, battery
        storage, and flexible natural-gas peaker plants to balance the
        intermittency of renewables. Smart grids allow two-way
        communication between utilities and consumers to shift load to
        times of abundant renewable supply.

        Policy Incentives: Many countries offer tax credits, feed-in
        tariffs, or renewable portfolio standards to accelerate adoption.
        The U.S. federal solar Investment Tax Credit (ITC), for example,
        allows homeowners and businesses to deduct a percentage of solar
        installation costs from their taxes.
        """,
    "space_exploration.txt": """
        A Brief Timeline of Space Exploration

        1957 - Sputnik 1: The Soviet Union launches the first artificial
        satellite, kicking off the Space Race with the United States.

        1969 - Apollo 11: NASA astronauts Neil Armstrong and Buzz Aldrin
        become the first humans to walk on the Moon, while Michael Collins
        orbits above in the command module.

        1981 - Space Shuttle: NASA's Space Shuttle Columbia becomes the
        first reusable crewed spacecraft, flying dozens of missions over
        three decades to deploy satellites and build the International
        Space Station.

        1998 - International Space Station (ISS): Construction begins on
        the ISS, a multi-nation collaboration that has hosted continuous
        human presence in low Earth orbit since November 2000.

        2012 - Commercial Cargo: SpaceX's Dragon capsule becomes the first
        commercial spacecraft to dock with the ISS, marking the start of
        commercial resupply missions.

        2020 - Crewed Commercial Flight: SpaceX's Crew Dragon carries NASA
        astronauts to the ISS, the first crewed orbital flight launched
        from U.S. soil since the Shuttle retired in 2011.

        2021-Present - Artemis Program: NASA's Artemis program aims to
        return humans to the lunar surface, including the first woman and
        person of color, and to establish a sustainable lunar presence as
        a stepping stone toward crewed Mars missions.
        """,
    "personal_finance.txt": """
        Personal Finance 101

        Budgeting: The 50/30/20 rule suggests allocating 50% of after-tax
        income to needs (housing, groceries, utilities), 30% to wants
        (dining out, entertainment), and 20% to savings and debt
        repayment. Tracking expenses for a month is the first step to
        building an accurate budget.

        Emergency Fund: Financial advisors typically recommend keeping 3
        to 6 months of essential living expenses in a readily accessible
        savings account before investing aggressively, to avoid going into
        debt when unexpected expenses arise.

        Compound Interest: Money invested early benefits from compound
        growth, where returns are reinvested and themselves earn returns.
        Starting to invest in your 20s instead of your 30s can roughly
        double the final retirement balance for the same monthly
        contribution, due to the extra decade of compounding.

        Debt Management: The "avalanche" method pays off debts with the
        highest interest rate first while making minimum payments on
        others, minimizing total interest paid. The "snowball" method pays
        off the smallest balance first for psychological motivation, even
        though it usually costs slightly more in total interest.

        Retirement Accounts: Employer-sponsored plans (like a 401(k) in
        the U.S.) often include matching contributions -- effectively free
        money -- so contributing at least enough to get the full match is
        usually recommended before investing elsewhere. Tax-advantaged
        individual accounts (IRAs) offer additional tax-deferred or
        tax-free growth depending on the account type.
        """,
}


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, body in DOCUMENTS.items():
        out_path = DOCS_DIR / filename
        out_path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
