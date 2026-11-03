"""The fictional insurer behind the eval corpus. Every fact in the golden sets starts here.

Ground truth is known by construction, because documents are rendered from this model. The sibling
plans are deliberately near-identical: same sentences, different numbers. That is what makes
retrieval hard in real insurance data, and why BM25 on the plan name alone is not enough.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

INSURER = "Tarnbrook Assurance"
NOTICE = "Fictional sample document for evaluation only."
SEED = 20261101
CANARY = "CANARY-7731"


@dataclass(frozen=True)
class Line:
    code: str
    name: str
    covers: str
    base_excess: int
    base_sum: int
    base_premium: int


LINES = [
    Line("DS", "DeviceShield", "phones, tablets and laptops", 1500, 80000, 2400),
    Line("HG", "HomeGuard", "home contents and fittings", 5000, 600000, 5200),
    Line("RS", "RoadSure", "private cars and two-wheelers", 3000, 900000, 7800),
    Line("VC", "VoyageCover", "domestic and international trips", 2000, 500000, 1900),
    Line("CF", "CareFirst", "hospitalisation and day-care treatment", 10000, 1000000, 11200),
    Line("AC", "ApplianceCare", "refrigerators, washers and air conditioners", 800, 90000, 1500),
]
TIERS = [("B", "Bronze", 1.6), ("S", "Silver", 1.3), ("G", "Gold", 1.0), ("P", "Platinum", 0.7)]


@dataclass(frozen=True)
class Plan:
    line: Line
    tier: str
    code: str
    excess: int
    sum_insured: int
    premium: int
    waiting_days: int
    free_look_days: int
    claim_window_days: int
    grace_days: int
    max_claims: int
    edition: int = 2026

    @property
    def name(self) -> str:
        return f"{self.line.name} {self.tier}"


@dataclass(frozen=True)
class Clause:
    """A hand-written policy clause with one fact and two ways of asking about it."""

    id: str
    line: str
    heading: str
    text: str
    quote: str
    direct_q: str
    paraphrase_q: str
    answer: str
    key_facts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Sop:
    id: str
    title: str
    purpose: str
    steps: tuple[str, ...]
    quote: str
    question: str
    answer: str
    key_facts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Note:
    """A short fact used for circulars, notices and meeting notes."""

    id: str
    title: str
    text: str
    quote: str
    question: str
    answer: str
    key_facts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    plan: Plan
    customer: str
    decision: str
    amount: int
    reason: str


@dataclass(frozen=True)
class Endorsement:
    number: str
    plan: Plan
    item: str
    old_value: str
    new_value: str
    effective: str


@dataclass
class World:
    plans: list[Plan] = field(default_factory=list)
    superseded: list[Plan] = field(default_factory=list)
    sublimits: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    premiums: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    service_levels: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    endorsements: list[Endorsement] = field(default_factory=list)


def _clause(
    id_: str,
    line: str,
    heading: str,
    text: str,
    quote: str,
    direct: str,
    para: str,
    answer: str,
    *facts: tuple[str, ...],
) -> Clause:
    return Clause(id_, line, heading, text, quote, direct, para, answer, tuple(facts))


CLAUSES = [
    # DeviceShield
    _clause(
        "ds-theft",
        "DS",
        "Theft and loss",
        "Theft is covered only where there is visible evidence of forced entry or violence. Loss "
        "of the insured device from an unlocked vehicle or an unattended public place is not "
        "covered, and neither is mysterious disappearance.",
        "Loss of the insured device from an unlocked vehicle or an unattended public place is "
        "not covered",
        "Does DeviceShield cover theft from an unattended public place?",
        "I left my laptop on a cafe table and walked off for a minute. It vanished. Will "
        "DeviceShield pay?",
        "No. Loss from an unattended public place is not covered.",
        ("not covered", "no cover", "isn't covered", "excluded", "not eligible"),
    ),
    _clause(
        "ds-battery",
        "DS",
        "Battery failure",
        "Battery swelling and capacity loss are covered only if maximum capacity has fallen below "
        "70 percent within the first 18 months of the device purchase date. Normal ageing after "
        "that period is treated as wear.",
        "maximum capacity has fallen below 70 percent within the first 18 months",
        "What is the battery capacity threshold under DeviceShield?",
        "My handset only holds about two thirds of its charge after a year. Can I get a new cell "
        "under DeviceShield?",
        "Yes, if capacity is below 70 percent within 18 months of purchase.",
        ("70", "seventy"),
        ("18", "eighteen"),
    ),
    _clause(
        "ds-liquid",
        "DS",
        "Liquid damage",
        "Accidental liquid damage is limited to two claims in any policy year. Corrosion that "
        "first appears more than 21 days after the incident is excluded because the cause can no "
        "longer be established.",
        "Corrosion that first appears more than 21 days after the incident is excluded",
        "How long after a liquid incident is corrosion excluded under DeviceShield?",
        "My tablet took a swim in the pool, and rust only showed up a month later. Is that still "
        "something DeviceShield pays for?",
        "No. Corrosion first appearing more than 21 days after the incident is excluded.",
        ("21", "twenty-one"),
    ),
    _clause(
        "ds-cosmetic",
        "DS",
        "Cosmetic damage",
        "Scratches, dents and other marks that do not affect how the device works are excluded, "
        "unless the Screen Plus add-on applies to the display.",
        "Scratches, dents and other marks that do not affect how the device works are excluded",
        "Are scratches covered under DeviceShield?",
        "There is a nick on my phone casing but everything still functions. Will DeviceShield "
        "fix how it looks?",
        "No. Marks that do not affect function are excluded.",
        ("excluded", "not covered", "isn't covered", "no cover"),
    ),
    _clause(
        "ds-salvage",
        "DS",
        "Replacement and salvage",
        "Where repair is uneconomic we replace the device with one of equal or lesser "
        "specification. The damaged device then becomes our property as salvage and must be "
        "handed over within 10 working days.",
        "The damaged device then becomes our property as salvage and must be handed over within "
        "10 working days",
        "What happens to the damaged device after a replacement under DeviceShield?",
        "If fixing my broken handset costs more than it's worth, can I hang on to the old one "
        "after DeviceShield gives me a new one?",
        "It becomes the insurer's salvage and must be handed over within 10 working days.",
        ("salvage", "property", "handed over", "hand it over", "hand over"),
        ("10", "ten"),
    ),
    # HomeGuard
    _clause(
        "hg-vacancy",
        "HG",
        "Unoccupied homes",
        "Cover for burst pipes and theft is suspended if the home is left unoccupied for more "
        "than 45 consecutive days. Fire and lightning cover continues during this period.",
        "suspended if the home is left unoccupied for more than 45 consecutive days",
        "When is HomeGuard cover suspended for unoccupied homes?",
        "We are going abroad for two months and nobody will be in the flat. If a water line "
        "bursts while we are gone, does HomeGuard respond?",
        "No. Burst-pipe cover is suspended after 45 consecutive unoccupied days.",
        ("45", "forty-five"),
    ),
    _clause(
        "hg-flood",
        "HG",
        "Flood events",
        "All flood loss within any 72 consecutive hours counts as one event, so a single excess "
        "applies. Water that returns after that window is a new event with a new excess.",
        "All flood loss within any 72 consecutive hours counts as one event",
        "How does the 72-hour clause work for HomeGuard flood claims?",
        "The river rose on Monday and came back on Wednesday morning, wrecking the ground floor "
        "twice. Do I pay my HomeGuard deductible once or twice?",
        "Once, because loss within 72 consecutive hours counts as one event.",
        ("72", "seventy-two"),
    ),
    _clause(
        "hg-jewellery",
        "HG",
        "Valuables",
        "Jewellery and watches are limited to 20 percent of the contents sum insured. Any single "
        "item worth more than Rs. 75,000 must be declared with a valuation certificate.",
        "Any single item worth more than Rs. 75,000 must be declared with a valuation certificate",
        "What is the declaration limit for jewellery under HomeGuard?",
        "My grandmother's necklace is worth about a lakh. Do I have to tell HomeGuard about it "
        "specifically?",
        "Yes. Single items above Rs. 75,000 must be declared with a valuation certificate.",
        ("75,000", "75000"),
    ),
    _clause(
        "hg-tenant",
        "HG",
        "Tenant damage",
        "Malicious damage by a tenant is covered only if a police report is filed within 48 hours "
        "of discovery. Unpaid rent is never covered.",
        "Malicious damage by a tenant is covered only if a police report is filed within 48 hours "
        "of discovery",
        "What is the police report deadline for tenant damage under HomeGuard?",
        "The renter smashed up my kitchen before moving out. How quickly must the authorities "
        "hear about it for HomeGuard to pay?",
        "A police report must be filed within 48 hours of discovery.",
        ("48", "forty-eight", "two days"),
    ),
    _clause(
        "hg-rent",
        "HG",
        "Alternative accommodation",
        "If an insured event makes the home uninhabitable, we pay for alternative accommodation "
        "for up to 90 days, subject to the limit in the schedule.",
        "we pay for alternative accommodation for up to 90 days",
        "How long is alternative accommodation paid under HomeGuard?",
        "A fire has left our house unlivable. For how long will HomeGuard fund a hotel for us?",
        "Up to 90 days.",
        ("90", "ninety", "three months"),
    ),
    # RoadSure
    _clause(
        "rs-licence",
        "RS",
        "Driver eligibility",
        "A claim is repudiated if the driver did not hold a valid licence at the time of the "
        "accident. A licence that expired no more than 30 days earlier and was then renewed is "
        "treated as valid.",
        "A licence that expired no more than 30 days earlier and was then renewed is treated as "
        "valid",
        "How does RoadSure treat an expired driving licence?",
        "My permit to drive lapsed three weeks before the crash, but I have since renewed it. "
        "Will RoadSure still honour the claim?",
        "Yes. A licence expired up to 30 days earlier and then renewed is treated as valid.",
        ("30", "thirty"),
    ),
    _clause(
        "rs-ncb",
        "RS",
        "No-claim bonus",
        "A single claim reduces the no-claim bonus by one slab. Two claims in one policy year "
        "reset the bonus to zero at renewal.",
        "Two claims in one policy year reset the bonus to zero at renewal",
        "How do two claims affect the RoadSure no-claim bonus?",
        "I have already had one fender bender this year. If I report a second, what happens to "
        "my discount for safe driving?",
        "It resets to zero at renewal.",
        ("zero", "reset", "0"),
    ),
    _clause(
        "rs-tow",
        "RS",
        "Towing",
        "Towing after an insured accident is reimbursed for up to 50 kilometres from the place of "
        "breakdown to the nearest approved garage.",
        "reimbursed for up to 50 kilometres from the place of breakdown",
        "What is the towing limit under RoadSure?",
        "My car died on the highway far from any workshop. How far will RoadSure pay for it to be "
        "dragged?",
        "Up to 50 kilometres.",
        ("50", "fifty"),
    ),
    _clause(
        "rs-flood",
        "RS",
        "Engine damage",
        "Engine damage caused by driving through standing water, including hydrostatic lock, is "
        "excluded unless the Engine Protect add-on was purchased before the loss.",
        "Engine damage caused by driving through standing water, including hydrostatic lock, is "
        "excluded unless the Engine Protect add-on was purchased",
        "Is hydrostatic lock covered under RoadSure?",
        "I tried crossing a waterlogged underpass and the motor seized. Will RoadSure pay?",
        "Only if the Engine Protect add-on was purchased before the loss.",
        ("Engine Protect",),
    ),
    _clause(
        "rs-zero",
        "RS",
        "Depreciation waiver",
        "The depreciation waiver applies to the first two claims in a policy year. Tyres and "
        "batteries always carry 50 percent depreciation.",
        "Tyres and batteries always carry 50 percent depreciation",
        "What depreciation applies to tyres under RoadSure?",
        "I bought the zero-dep option. If my rubber and battery get damaged, do I get paid in "
        "full by RoadSure?",
        "No. Tyres and batteries always carry 50 percent depreciation.",
        ("50", "fifty", "half"),
    ),
    # VoyageCover
    _clause(
        "vc-delay",
        "VC",
        "Trip delay",
        "The trip delay benefit starts after 6 hours and is paid for each further complete block "
        "of 6 hours, up to the limit in the schedule. Delays announced before departure from "
        "home are excluded.",
        "The trip delay benefit starts after 6 hours",
        "When does VoyageCover trip delay start?",
        "My flight is stuck on the tarmac and has been pushed back again. How long must I wait "
        "before VoyageCover starts paying?",
        "After 6 hours.",
        ("6 hours", "six hours"),
    ),
    _clause(
        "vc-baggage",
        "VC",
        "Baggage delay",
        "Baggage delay pays only if bags are delayed beyond 12 hours on the outbound journey. "
        "Delays on the return leg are excluded because you are already home.",
        "Delays on the return leg are excluded",
        "Are return-leg baggage delays covered by VoyageCover?",
        "My suitcase got lost on the flight back to Delhi and turned up two days later. Can "
        "VoyageCover reimburse me?",
        "No. Delays on the return leg are excluded.",
        ("excluded", "not covered", "isn't covered", "outbound", "no cover"),
    ),
    _clause(
        "vc-adventure",
        "VC",
        "Adventure activities",
        "Scuba diving below 30 metres, paragliding and any activity done without a licensed "
        "guide are excluded from medical and accident cover.",
        "Scuba diving below 30 metres, paragliding and any activity done without a licensed guide "
        "are excluded",
        "How deep can I scuba dive under VoyageCover?",
        "I plan a deep wreck dive on holiday, down to 35 m. Am I protected if something goes "
        "wrong, under VoyageCover?",
        "No. Diving below 30 metres is excluded.",
        ("30", "thirty"),
    ),
    _clause(
        "vc-ped",
        "VC",
        "Existing conditions",
        "Emergency treatment of a pre-existing condition abroad is capped at 10 percent of the "
        "medical sum insured. Planned treatment is never covered.",
        "Emergency treatment of a pre-existing condition abroad is capped at 10 percent of the "
        "medical sum insured",
        "What is the VoyageCover cap on pre-existing conditions?",
        "I have long-standing diabetes and had a crisis in Singapore. How much of my medical "
        "limit can VoyageCover use for that?",
        "Up to 10 percent of the medical sum insured.",
        ("10 percent", "10%", "ten percent"),
    ),
    _clause(
        "vc-cancel",
        "VC",
        "Trip cancellation",
        "Trip cancellation pays only for the death or hospitalisation of an insured person or an "
        "immediate family member. Visa refusal and change of mind are excluded.",
        "Visa refusal and change of mind are excluded",
        "Is visa refusal covered by VoyageCover cancellation?",
        "The consulate turned down my application so I can't fly. Will VoyageCover refund my trip?",
        "No. Visa refusal is excluded.",
        ("excluded", "not covered", "isn't covered", "no cover"),
    ),
    # CareFirst
    _clause(
        "cf-ped",
        "CF",
        "Pre-existing diseases",
        "Pre-existing diseases are covered after 36 months of continuous cover with us. Porting "
        "from another insurer counts towards this period if there was no break.",
        "Pre-existing diseases are covered after 36 months of continuous cover",
        "What is the CareFirst waiting period for pre-existing diseases?",
        "I've had high blood pressure since before I joined. How many years until CareFirst pays "
        "for related hospital stays?",
        "After 36 months, which is three years.",
        ("36", "three years", "3 years"),
    ),
    _clause(
        "cf-room",
        "CF",
        "Room rent",
        "Room rent is capped at 1 percent of the sum insured per day. A higher room category "
        "reduces every other bill item in the same proportion.",
        "A higher room category reduces every other bill item in the same proportion",
        "What happens if I choose a room above the CareFirst room rent cap?",
        "I picked a deluxe suite above my entitlement. Does that only affect the room charge "
        "under CareFirst?",
        "No. Every other bill item is reduced in the same proportion.",
        ("proportion", "proportionate", "every other", "all other"),
    ),
    _clause(
        "cf-cataract",
        "CF",
        "Sub-limits",
        "Cataract surgery is limited to Rs. 40,000 per eye in any policy year, regardless of the "
        "sum insured.",
        "Cataract surgery is limited to Rs. 40,000 per eye",
        "What is the cataract limit under CareFirst?",
        "My father needs a lens replaced because of clouded vision. What is the most CareFirst "
        "will pay for each side?",
        "Rs. 40,000 per eye.",
        ("40,000", "40000"),
    ),
    _clause(
        "cf-maternity",
        "CF",
        "Maternity",
        "Maternity expenses are covered after a waiting period of 24 months. Complications of "
        "pregnancy follow the same waiting period.",
        "Maternity expenses are covered after a waiting period of 24 months",
        "What is the CareFirst maternity waiting period?",
        "We are planning a baby next year and bought CareFirst recently. When does delivery "
        "become claimable?",
        "After 24 months.",
        ("24", "two years", "2 years"),
    ),
    _clause(
        "cf-cashless",
        "CF",
        "Cashless admission",
        "Cashless requests for planned admissions must reach the claims team 72 hours before "
        "admission. For an emergency, the request must arrive within 24 hours of admission.",
        "Cashless requests for planned admissions must reach the claims team 72 hours before "
        "admission",
        "How early must a planned CareFirst cashless request be sent?",
        "My knee surgery is scheduled for next month. How much notice does CareFirst need so that "
        "I don't pay upfront?",
        "72 hours before admission.",
        ("72", "seventy-two", "three days"),
    ),
    # ApplianceCare
    _clause(
        "ac-compressor",
        "AC",
        "Compressor cover",
        "Compressor cover on refrigerators and air conditioners runs for 5 years from purchase. "
        "All other parts are covered for 2 years.",
        "Compressor cover on refrigerators and air conditioners runs for 5 years from purchase",
        "How long is compressor cover under ApplianceCare?",
        "My fridge is four years old and has stopped cooling because its motor unit failed. Is "
        "that still protected by ApplianceCare?",
        "Yes. Compressor cover runs for 5 years from purchase.",
        ("5 years", "five years"),
    ),
    _clause(
        "ac-voltage",
        "AC",
        "Electrical damage",
        "Damage from voltage fluctuation is covered only if a surge protector or stabiliser "
        "approved by the manufacturer was installed at the time.",
        "Damage from voltage fluctuation is covered only if a surge protector or stabiliser "
        "approved by the manufacturer was installed",
        "Is voltage fluctuation damage covered by ApplianceCare?",
        "A power spike fried my washing machine. What would I need to have had in place for "
        "ApplianceCare to pay?",
        "A manufacturer-approved surge protector or stabiliser.",
        ("stabiliser", "stabilizer", "surge protector"),
    ),
    _clause(
        "ac-consumables",
        "AC",
        "Consumables",
        "Filters, refrigerant gas top-ups and remote control batteries are consumables and are "
        "excluded, even when replaced during a covered repair.",
        "Filters, refrigerant gas top-ups and remote control batteries are consumables and are "
        "excluded",
        "Are filters covered under ApplianceCare?",
        "The engineer who fixed my AC also refilled the coolant. Do I pay for the coolant myself "
        "under ApplianceCare?",
        "Yes. Refrigerant gas top-ups are consumables and are excluded.",
        ("consumable", "excluded", "not covered", "isn't covered"),
    ),
    _clause(
        "ac-visit",
        "AC",
        "Service visits",
        "An engineer visits within 48 hours in metro cities and within 96 hours elsewhere, "
        "counted from when the service request is accepted.",
        "An engineer visits within 48 hours in metro cities and within 96 hours elsewhere",
        "What is the ApplianceCare engineer visit time outside metros?",
        "I live in a small town and my washer has broken. How soon should ApplianceCare send "
        "somebody?",
        "Within 96 hours outside metro cities.",
        ("96", "four days", "ninety-six"),
    ),
    _clause(
        "ac-relocation",
        "AC",
        "Relocation",
        "Cover ends if the appliance is moved by anyone other than an authorised installer, "
        "unless the move is reported within 15 days and the appliance passes an inspection.",
        "unless the move is reported within 15 days and the appliance passes an inspection",
        "How can ApplianceCare cover survive a relocation?",
        "We shifted house and our own movers carried the split AC. Can ApplianceCare still stay "
        "in force?",
        "Yes, if the move is reported within 15 days and the appliance passes an inspection.",
        ("15", "fifteen"),
    ),
]

SOPS = [
    Sop(
        "sop-siu",
        "Special Investigations Referral",
        "When a claim must go to the Special Investigations Unit.",
        (
            "Check the policy age and the claimed amount.",
            "Refer any claim above Rs. 2,00,000 on a policy younger than 60 days to the Special "
            "Investigations Unit.",
            "Record the referral in the claim file and pause the payment timer.",
        ),
        "Refer any claim above Rs. 2,00,000 on a policy younger than 60 days",
        "Which claims go to the Special Investigations Unit?",
        "Claims above Rs. 2,00,000 on a policy younger than 60 days.",
        (("2,00,000", "200000", "2 lakh", "200,000"), ("60", "sixty")),
    ),
    Sop(
        "sop-refund",
        "Premium Refund Approval",
        "Who can approve a premium refund.",
        (
            "Verify that the cancellation request is signed.",
            "A team lead may approve refunds up to Rs. 25,000; anything larger needs the regional "
            "finance controller.",
            "Pay the refund to the original payment method only.",
        ),
        "A team lead may approve refunds up to Rs. 25,000",
        "Who approves a premium refund larger than Rs. 25,000?",
        "The regional finance controller.",
        (("regional finance controller", "finance controller"),),
    ),
    Sop(
        "sop-grievance",
        "Customer Grievance Handling",
        "How complaints are acknowledged.",
        (
            "Log the complaint in the grievance register.",
            "Acknowledge every grievance within 3 working days and close it within 14 calendar "
            "days.",
            "Escalate unresolved items to the Grievance Officer.",
        ),
        "Acknowledge every grievance within 3 working days and close it within 14 calendar days",
        "How quickly must a customer grievance be closed?",
        "Within 14 calendar days.",
        (("14", "fourteen"),),
    ),
    Sop(
        "sop-survey",
        "Surveyor Appointment",
        "When a licensed surveyor is required.",
        (
            "Estimate the loss from the claim form.",
            "Appoint a licensed surveyor for any motor loss estimated above Rs. 50,000.",
            "Share the survey report with the customer on request.",
        ),
        "Appoint a licensed surveyor for any motor loss estimated above Rs. 50,000",
        "At what estimated motor loss must a licensed surveyor be appointed?",
        "Above Rs. 50,000.",
        (("50,000", "50000"),),
    ),
    Sop(
        "sop-kyc",
        "KYC Re-verification",
        "When to repeat KYC checks.",
        (
            "Check the date of the last KYC record.",
            "Repeat KYC before paying any claim if the record is older than 8 years.",
            "Use only the approved verification vendors.",
        ),
        "Repeat KYC before paying any claim if the record is older than 8 years",
        "How old can a KYC record be before it has to be repeated for a claim payment?",
        "8 years.",
        (("8 years", "eight years"),),
    ),
    Sop(
        "sop-retention",
        "Claim Document Retention",
        "How long claim files are kept.",
        (
            "Archive the file when the claim closes.",
            "Keep closed claim files for 10 years, then destroy them under dual sign-off.",
            "Litigation files are kept until the case ends.",
        ),
        "Keep closed claim files for 10 years, then destroy them under dual sign-off",
        "How long are closed claim files retained?",
        "10 years.",
        (("10 years", "ten years"),),
    ),
    Sop(
        "sop-reopen",
        "Reopening a Closed Claim",
        "When a closed claim may be reopened.",
        (
            "Confirm that new evidence exists.",
            "A closed claim may be reopened once within 12 months if new evidence appears.",
            "Assign the file to a different handler.",
        ),
        "A closed claim may be reopened once within 12 months if new evidence appears",
        "How many times and for how long can a closed claim be reopened?",
        "Once, within 12 months.",
        (("once", "one time"), ("12 months", "twelve months", "one year", "1 year")),
    ),
    Sop(
        "sop-exgratia",
        "Ex-gratia Payments",
        "Goodwill payments outside policy terms.",
        (
            "Record why the claim is outside the policy terms.",
            "Ex-gratia payments are capped at Rs. 15,000 and need the Head of Claims to sign.",
            "Report all ex-gratia payments to the board each quarter.",
        ),
        "Ex-gratia payments are capped at Rs. 15,000",
        "What is the cap on ex-gratia payments?",
        "Rs. 15,000.",
        (("15,000", "15000"),),
    ),
    Sop(
        "sop-network",
        "Cashless Hospital Delisting",
        "How hospitals leave the network.",
        (
            "Review the hospital's claims history.",
            "Give affected members 30 days of notice before a hospital is delisted.",
            "Update the network list the same day.",
        ),
        "Give affected members 30 days of notice before a hospital is delisted",
        "How much notice do members get before a hospital is delisted?",
        "30 days.",
        (("30", "thirty"),),
    ),
    Sop(
        "sop-death",
        "Death Claim Priority",
        "Fast handling of death claims.",
        (
            "Assign a senior handler on the day of intimation.",
            "Death claims receive a first contact within 2 hours of intimation.",
            "Waive the courier requirement for original documents.",
        ),
        "Death claims receive a first contact within 2 hours of intimation",
        "How fast must first contact happen on a death claim?",
        "Within 2 hours of intimation.",
        (("2 hours", "two hours"),),
    ),
    Sop(
        "sop-subrogation",
        "Subrogation Recovery",
        "Recovering paid claims from third parties.",
        (
            "Identify the third party responsible.",
            "Open recovery on every paid third-party-fault claim above Rs. 1,00,000.",
            "Send the recovery notice by registered post.",
        ),
        "Open recovery on every paid third-party-fault claim above Rs. 1,00,000",
        "Above what amount must recovery be opened on third-party-fault claims?",
        "Above Rs. 1,00,000.",
        (("1,00,000", "100000", "1 lakh", "100,000"),),
    ),
    Sop(
        "sop-fraud",
        "Staff Fraud Reporting",
        "How employees report suspected fraud.",
        (
            "Do not confront the person concerned.",
            "Report suspected staff fraud to the ethics line within 24 hours.",
            "Preserve all related emails and files.",
        ),
        "Report suspected staff fraud to the ethics line within 24 hours",
        "How soon must suspected staff fraud be reported?",
        "Within 24 hours, to the ethics line.",
        (("24", "twenty-four"),),
    ),
    Sop(
        "sop-vendor",
        "Repair Vendor Audit",
        "How approved repair vendors are audited.",
        (
            "Sample completed repairs each month.",
            "Audit each approved repair vendor twice a year and suspend any scoring below 75 "
            "points.",
            "Publish the scorecard to the vendor.",
        ),
        "Audit each approved repair vendor twice a year and suspend any scoring below 75 points",
        "What audit score gets a repair vendor suspended?",
        "Below 75 points.",
        (("75", "seventy-five"),),
    ),
    Sop(
        "sop-language",
        "Regional Language Support",
        "Serving customers in their own language.",
        (
            "Record the customer's preferred language.",
            "Offer regional language support in 11 languages during business hours.",
            "Translate decision letters on request.",
        ),
        "Offer regional language support in 11 languages",
        "In how many languages is regional support offered?",
        "11 languages.",
        (("11", "eleven"),),
    ),
    Sop(
        "sop-data",
        "Customer Data Requests",
        "Handling requests to access personal data.",
        (
            "Verify the identity of the requester.",
            "Respond to a personal data access request within 20 working days.",
            "Log the response in the privacy register.",
        ),
        "Respond to a personal data access request within 20 working days",
        "How long do we have to answer a personal data access request?",
        "Within 20 working days.",
        (("20", "twenty"),),
    ),
    Sop(
        "sop-catastrophe",
        "Catastrophe Response",
        "Claims handling after a declared catastrophe.",
        (
            "Activate the catastrophe desk.",
            "During a declared catastrophe, raise handler payment authority to Rs. 3,00,000.",
            "Send daily loss counts to the reinsurer.",
        ),
        "raise handler payment authority to Rs. 3,00,000",
        "What does handler payment authority rise to during a declared catastrophe?",
        "Rs. 3,00,000.",
        (("3,00,000", "300000", "3 lakh", "300,000"),),
    ),
]

CIRCULARS = [
    Note(
        "cir-digital",
        "Digital Policy Copies",
        "From 1 April 2026 every policy is issued electronically unless the customer asks for "
        "paper. A paper copy costs Rs. 150.",
        "A paper copy costs Rs. 150",
        "How much does a paper policy copy cost?",
        "Rs. 150.",
        (("150",),),
    ),
    Note(
        "cir-nominee",
        "Nominee Updates",
        "Nominee changes now take effect on the date of the customer's request, not the date of "
        "processing, provided the request is signed.",
        "Nominee changes now take effect on the date of the customer's request",
        "When does a nominee change take effect?",
        "On the date of the customer's request.",
        (
            (
                "date of the customer's request",
                "date of request",
                "date of the request",
                "customer's request",
            ),
        ),
    ),
    Note(
        "cir-upi",
        "Claim Payments by UPI",
        "Claims up to Rs. 40,000 can be paid by UPI. Larger amounts go by bank transfer.",
        "Claims up to Rs. 40,000 can be paid by UPI",
        "What is the maximum claim that can be paid by UPI?",
        "Rs. 40,000.",
        (("40,000", "40000"),),
    ),
    Note(
        "cir-senior",
        "Senior Citizen Helpline",
        "Customers above 60 reach a dedicated helpline on 1800-555-0142 from 8 am to 8 pm.",
        "Customers above 60 reach a dedicated helpline on 1800-555-0142",
        "What is the helpline number for senior citizens?",
        "1800-555-0142.",
        (("1800-555-0142", "18005550142"),),
    ),
    Note(
        "cir-whatsapp",
        "WhatsApp Claims Intake",
        "Device and appliance claims may be started on WhatsApp. Motor and health claims may "
        "not, because a signed form is required.",
        "Motor and health claims may not",
        "Can a health claim be started on WhatsApp?",
        "No. Motor and health claims may not, because a signed form is required.",
        (("may not", "cannot", "can't", "not be", "no"),),
    ),
    Note(
        "cir-gst",
        "GST on Premiums",
        "Premium refunds now include the GST originally collected, shown as a separate line on "
        "the credit note.",
        "Premium refunds now include the GST originally collected",
        "Do premium refunds include GST?",
        "Yes. They include the GST originally collected.",
        (("GST",),),
    ),
    Note(
        "cir-ev",
        "Electric Vehicle Batteries",
        "Traction batteries of electric cars are covered under RoadSure only up to 8 years from "
        "first registration.",
        "Traction batteries of electric cars are covered under RoadSure only up to 8 years",
        "How long are electric car traction batteries covered under RoadSure?",
        "Up to 8 years from first registration.",
        (("8 years", "eight years"),),
    ),
    Note(
        "cir-tele",
        "Teleconsultation Benefit",
        "CareFirst members receive 6 free teleconsultations in each policy year.",
        "CareFirst members receive 6 free teleconsultations",
        "How many free teleconsultations do CareFirst members get?",
        "6 per policy year.",
        (("6", "six"),),
    ),
    Note(
        "cir-sanction",
        "Sanctioned Destinations",
        "VoyageCover cannot be issued for travel to a destination under a national travel "
        "advisory of level 4.",
        "under a national travel advisory of level 4",
        "Which destinations can't VoyageCover be issued for?",
        "Destinations under a national travel advisory of level 4.",
        (("level 4", "level four"),),
    ),
    Note(
        "cir-cooling",
        "Free-look Refund Deductions",
        "A free-look refund deducts stamp duty and a proportionate risk premium, and nothing else.",
        "A free-look refund deducts stamp duty and a proportionate risk premium",
        "What is deducted from a free-look refund?",
        "Stamp duty and a proportionate risk premium.",
        (("stamp duty",), ("risk premium",)),
    ),
    Note(
        "cir-email",
        "Claim Status Emails",
        "Customers receive an automatic claim status email every 5 working days until closure.",
        "automatic claim status email every 5 working days",
        "How often do customers get a claim status email?",
        "Every 5 working days.",
        (("5 working days", "five working days"),),
    ),
    Note(
        "cir-age",
        "Upper Age Limit",
        "New CareFirst proposals are accepted up to age 70. Existing members renew for life.",
        "New CareFirst proposals are accepted up to age 70",
        "What is the entry age limit for CareFirst?",
        "Up to age 70.",
        (("70", "seventy"),),
    ),
]

MEETING_NOTES = [
    Note(
        "note-pilot",
        "Claims ops weekly",
        "Decided: the photo-based device triage pilot runs in Pune and Jaipur until the end of "
        "February.",
        "photo-based device triage pilot runs in Pune and Jaipur",
        "Which cities are in the photo-based device triage pilot?",
        "Pune and Jaipur.",
        (("Pune",), ("Jaipur",)),
    ),
    Note(
        "note-owner",
        "Vendor review",
        "Meera will own the courier contract renewal; the target rate is Rs. 38 per kg.",
        "target rate is Rs. 38 per kg",
        "What target rate was agreed for the courier contract?",
        "Rs. 38 per kg.",
        (("38",),),
    ),
    Note(
        "note-freeze",
        "Release planning",
        "No changes to the claims portal between 24 and 31 March because of year-end closing.",
        "between 24 and 31 March",
        "When is the claims portal change freeze?",
        "Between 24 and 31 March.",
        (("24",), ("31",)),
    ),
    Note(
        "note-sla",
        "Monsoon readiness",
        "For the monsoon we agreed an extra 40 surveyors on call in Mumbai.",
        "an extra 40 surveyors on call in Mumbai",
        "How many extra surveyors are on call in Mumbai for the monsoon?",
        "40.",
        (("40", "forty"),),
    ),
    Note(
        "note-budget",
        "Quarterly finance sync",
        "The chatbot budget was cut from Rs. 18 lakh to Rs. 11 lakh for next quarter.",
        "cut from Rs. 18 lakh to Rs. 11 lakh",
        "What is the chatbot budget for next quarter?",
        "Rs. 11 lakh.",
        (("11 lakh",),),
    ),
    Note(
        "note-train",
        "Training committee",
        "All claims handlers must finish the fraud awareness module by 15 May.",
        "finish the fraud awareness module by 15 May",
        "What is the deadline for the fraud awareness module?",
        "15 May.",
        (("15 May", "May 15"),),
    ),
    Note(
        "note-tat",
        "Service review",
        "Average device claim turnaround hit 3.4 days last month; the goal stays at 3 days.",
        "turnaround hit 3.4 days last month",
        "What was the average device claim turnaround last month?",
        "3.4 days.",
        (("3.4",),),
    ),
    Note(
        "note-hire",
        "Hiring update",
        "We approved two more actuarial analysts for the Bengaluru office.",
        "two more actuarial analysts for the Bengaluru office",
        "Where are the two new actuarial analysts based?",
        "The Bengaluru office.",
        (("Bengaluru", "Bangalore"),),
    ),
    Note(
        "note-escal",
        "Escalations forum",
        "Repeat complaints from the same customer now go straight to level 2 support.",
        "go straight to level 2 support",
        "Where do repeat complaints go now?",
        "Straight to level 2 support.",
        (("level 2", "level two"),),
    ),
    Note(
        "note-tpa",
        "Health desk catch-up",
        "The third-party administrator contract ends in September and will not be renewed.",
        "contract ends in September and will not be renewed",
        "When does the third-party administrator contract end?",
        "September, with no renewal.",
        (("September",),),
    ),
]

NOTICES = [
    Note(
        "ntc-office",
        "Office Relocation Notice",
        "Our Nagpur branch moves to Sitabuldi on 3 August.",
        "Our Nagpur branch moves to Sitabuldi on 3 August",
        "When does the Nagpur branch move?",
        "On 3 August.",
        (("3 August", "August 3"),),
    ),
    Note(
        "ntc-hours",
        "Revised Counter Hours",
        "Claims counters close at 4:30 pm on the second Saturday of each month.",
        "Claims counters close at 4:30 pm on the second Saturday",
        "When do claims counters close on the second Saturday?",
        "At 4:30 pm.",
        (("4:30", "4.30"),),
    ),
    Note(
        "ntc-phish",
        "Phishing Warning",
        "We never ask for an OTP over the phone. Report such calls to 1800-555-0177.",
        "Report such calls to 1800-555-0177",
        "Which number do I call to report a suspicious OTP request?",
        "1800-555-0177.",
        (("1800-555-0177", "18005550177"),),
    ),
    Note(
        "ntc-camp",
        "Health Check Camp",
        "A free health check camp runs at the Indore office on 12 October.",
        "A free health check camp runs at the Indore office on 12 October",
        "Where and when is the free health check camp?",
        "At the Indore office on 12 October.",
        (("Indore",), ("12 October", "October 12")),
    ),
    Note(
        "ntc-maint",
        "Portal Maintenance",
        "The customer portal is offline from 1 am to 5 am this Sunday.",
        "offline from 1 am to 5 am this Sunday",
        "When is the customer portal offline?",
        "From 1 am to 5 am on Sunday.",
        (("1 am",), ("5 am",)),
    ),
    Note(
        "ntc-bank",
        "Change of Collection Account",
        "Premium collections move to account ending 4471 from the next billing cycle.",
        "account ending 4471",
        "Which account do premium collections move to?",
        "The account ending 4471.",
        (("4471",),),
    ),
    Note(
        "ntc-holiday",
        "Holiday Notice",
        "Offices stay closed on 14 April for the regional holiday.",
        "Offices stay closed on 14 April",
        "Which day are offices closed for the regional holiday?",
        "14 April.",
        (("14 April", "April 14"),),
    ),
    Note(
        "ntc-courier",
        "Courier Change",
        "Original bills must now be sent by Speed Post only.",
        "Original bills must now be sent by Speed Post only",
        "How must original bills be sent?",
        "By Speed Post only.",
        (("Speed Post",),),
    ),
]

GENERAL_CONDITIONS = [
    (
        "Duty of disclosure",
        "You must give true and complete answers in the proposal. If a material fact was hidden, "
        "we "
        "may treat the policy as void from inception and retain the premium paid.",
    ),
    (
        "Other insurance",
        "If another policy covers the same loss, we pay only our rateable share. You must tell us "
        "about every such policy when you make a claim.",
    ),
    (
        "Fraud",
        "If any claim is fraudulent in any respect, all benefit under this policy is forfeited and "
        "we may recover any amount already paid.",
    ),
    (
        "Arbitration",
        "A dispute about the amount payable goes to a sole arbitrator under the Arbitration and "
        "Conciliation Act, while a dispute about liability goes to the courts of India.",
    ),
    (
        "Notices",
        "Any notice from us is treated as received 3 days after posting to the last address you "
        "gave us. Notices to us must be in writing.",
    ),
    (
        "Assignment",
        "This policy may not be assigned without our written consent, except on the sale of the "
        "insured property with due notice.",
    ),
]

DEFINITIONS = [
    ("Accident", "A sudden, unforeseen and involuntary event caused by external means."),
    ("Excess", "The amount you bear on each claim before we make any payment."),
    ("Policy year", "A period of twelve months from the start date or any renewal date."),
    ("Sum insured", "The maximum amount we pay in a policy year, as shown in the schedule."),
    ("Wear and tear", "Gradual loss of value or function through normal use."),
]

SUBLIMIT_NAMES = {
    "DS": ["Screen repair", "Accessories", "Data recovery", "Loaner device per day"],
    "HG": [
        "Contents in transit",
        "Garden equipment",
        "Domestic staff belongings",
        "Locks and keys",
    ],
    "RS": ["Personal effects", "Child seat", "Roadside fuel delivery", "Key replacement"],
    "VC": ["Passport loss", "Hotel extension", "Emergency cash", "Missed connection"],
    "CF": ["Ambulance per event", "Home nursing", "Physiotherapy", "Second opinion"],
    "AC": ["Control board", "Drum or tub", "Gas charging labour", "Stand or bracket"],
}
AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65"]
SLA_NAMES = [
    "First response (hours)",
    "Document review (days)",
    "Payment after approval (days)",
    "Complaint callback (hours)",
]


def _round_to(value: float, step: int) -> int:
    return int(round(value / step) * step)


def _make_plan(rng: random.Random, line: Line, tier: tuple[str, str, float], edition: int) -> Plan:
    tier_code, tier_name, factor = tier
    sum_factor = 2.2 - factor
    return Plan(
        line=line,
        tier=tier_name,
        code=f"{line.code}-{tier_code}-{rng.randint(1000, 9999)}",
        excess=_round_to(line.base_excess * factor + rng.randint(-120, 120), 50),
        sum_insured=_round_to(line.base_sum * sum_factor, 5000),
        premium=_round_to(line.base_premium * sum_factor + rng.randint(-90, 90), 10),
        waiting_days=rng.choice([7, 14, 21, 30, 45]),
        free_look_days=rng.choice([15, 30]),
        claim_window_days=rng.choice([7, 10, 14, 21, 30, 45]),
        grace_days=rng.choice([15, 21, 30]),
        max_claims=rng.choice([1, 2, 3, 4, 5]),
        edition=edition,
    )


def build_world() -> World:
    rng = random.Random(SEED)
    world = World()
    for line in LINES:
        for tier in TIERS:
            world.plans.append(_make_plan(rng, line, tier, 2026))

    # Old editions keep the plan code but state different facts, so a stale answer is detectable.
    for plan in rng.sample(world.plans, 10):
        world.superseded.append(
            Plan(
                line=plan.line,
                tier=plan.tier,
                code=plan.code,
                excess=plan.excess + rng.choice([250, 400, 500]),
                sum_insured=plan.sum_insured,
                premium=plan.premium - rng.choice([100, 150]),
                waiting_days=plan.waiting_days + rng.choice([5, 10]),
                free_look_days=plan.free_look_days,
                claim_window_days=plan.claim_window_days + rng.choice([2, 3]),
                grace_days=plan.grace_days,
                max_claims=plan.max_claims,
                edition=2025,
            )
        )

    for line in LINES:
        tiers = [name for _, name, _ in TIERS]
        world.sublimits[line.code] = {
            name: {
                tier: _round_to(
                    line.base_excess * (i + 1) * (j + 2) * 0.9 + rng.randint(0, 400), 50
                )
                for j, tier in enumerate(tiers)
            }
            for i, name in enumerate(SUBLIMIT_NAMES[line.code])
        }
        world.premiums[line.code] = {
            band: {
                tier: _round_to(
                    line.base_premium * (1 + 0.28 * i) * (1 + 0.21 * j) + rng.randint(-60, 60), 5
                )
                for j, tier in enumerate(tiers)
            }
            for i, band in enumerate(AGE_BANDS)
        }
        world.service_levels[line.code] = {
            name: {
                tier: str(rng.randint(2, 9) * (4 if "hours" in name else 1) + 3 - j)
                for j, tier in enumerate(tiers)
            }
            for name in SLA_NAMES
        }

    customers = [
        "Asha Raman",
        "Vikram Sethi",
        "Nisha Kulkarni",
        "Rohan Mehra",
        "Farah Qureshi",
        "Dev Malhotra",
        "Lata Iyer",
        "Sameer Bose",
        "Pooja Nair",
        "Arjun Ghosh",
    ]
    outcomes = [
        ("approved", "the loss falls within cover and all documents were complete"),
        ("approved in part", "the depreciation schedule reduced the assessed amount"),
        ("declined", "the loss was reported after the notification window closed"),
        ("declined", "the event falls under a listed exclusion in the policy wording"),
        ("on hold", "the repair invoice is missing and must be supplied"),
    ]
    used_ids: set[int] = set()
    for i in range(30):
        claim_number = rng.randint(40000, 99999)
        while claim_number in used_ids:
            claim_number = rng.randint(40000, 99999)
        used_ids.add(claim_number)
        decision, reason = outcomes[i % len(outcomes)]
        world.claims.append(
            Claim(
                claim_id=f"CLM-{claim_number}",
                plan=rng.choice(world.plans),
                customer=customers[i % len(customers)],
                decision=decision,
                amount=_round_to(rng.randint(4000, 90000), 100),
                reason=reason,
            )
        )

    items = [
        ("accidental damage sub-limit", "per claim"),
        ("loaner benefit", "per claim"),
        ("courier reimbursement", "per claim"),
        ("inspection fee waiver", "per policy year"),
    ]
    for i in range(28):
        plan = world.plans[(i * 5) % len(world.plans)]
        item, unit = items[i % len(items)]
        old = _round_to(rng.randint(5000, 20000), 500)
        new = old + _round_to(rng.randint(1000, 9000), 500)
        world.endorsements.append(
            Endorsement(
                number=f"E-{100 + i}",
                plan=plan,
                item=item,
                old_value=f"Rs. {old:,} {unit}",
                new_value=f"Rs. {new:,} {unit}",
                effective=f"{(i % 27) + 1} June 2026",
            )
        )
    return world


def indian_format(amount: int) -> str:
    """1500000 becomes 15,00,000, the grouping used in Indian documents."""
    digits = str(amount)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def amount_variants(amount: int) -> tuple[str, ...]:
    """Ways an answer may legitimately write the same amount."""
    variants = {indian_format(amount), f"{amount:,}", str(amount)}
    if amount >= 100000 and amount % 10000 == 0:
        lakh = amount / 100000
        variants.add(f"{lakh:g} lakh")
    return tuple(sorted(variants))
