"""Detector v2 (src/api/school_scope.py), shared by C1 and C2: the unit tests of results/round2/other_school_v2_plan_draft.md
section 10 (true positives, false positives, uncertain identification, C1 probes) and C2 behaviour on top of it."""
from types import SimpleNamespace as NS

import pytest

from src.api.school_scope import detect, mentions, notice
from src.verification import other_school as osc
from src.verification.verifier import VerificationResult


# ---- true positives -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, names", [
    ("Philippine Science High School suspended in-person classes on September 10.", ["Philippine Science High School"]),
    ("PHILIPPINE SCIENCE HIGH SCHOOL SUSPENDED CLASSES", ["PHILIPPINE SCIENCE HIGH SCHOOL"]),
    ("CLASSES AT PHILIPPINE SCIENCE HIGH SCHOOL ARE SUSPENDED", ["PHILIPPINE SCIENCE HIGH SCHOOL"]),
    ("Hogwarts Academy achieved a 99.35% passing rate", ["Hogwarts Academy"]),
    ("Did Holy Spirit Academy win?", ["Holy Spirit Academy"]),
    ("Springfield University students", ["Springfield University"]),
    ("Students from the University of Electro-Communications came", ["University of Electro-Communications"]),
    ("Seoul National University students visited", ["Seoul National University"]),
    ("National University won the UAAP title", ["National University"]),                  # the one generic-name exception
    ("The passing rate of STI College was 99.35%.", ["STI"]),                               # list (v1) kept
    ("Did La Salle hold a Teachers' Day celebration?", ["De La Salle"]),
    ("CLASSES AT BULACAN STATE UNIVERSITY ARE SUSPENDED", ["Bulacan State University"]),     # any-case alias in all caps
    ("Bulacan State University and Philippine Science High School suspended classes",
     ["Bulacan State University", "Philippine Science High School"]),
])
def test_true_positives(text, names):
    assert detect(text) == names


# ---- false positives (must not detect) --------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "The University will hold the midterm examinations on September 28 to 30.",              # generic (release-review failure)
    "Our University foundation day is on October 20.", "State University students joined", "our national university is great",
    "THE MIDTERM EXAMINATIONS FOR THE 1ST SEMESTER ARE SCHEDULED ON SEPTEMBER 28 UP TO 30, 2026.",   # all caps "UP"
    "AMA KO SAID WALANG PASOK BUKAS", "STI LL NO CLASSES TOMORROW AT ALL LEVELS",
    "IN-PERSON CLASSES AT ALL LEVELS WERE SUSPENDED ON SEPTEMBER 10 BECAUSE OF THE SOUTHWEST MONSOON.",
    "The College of Health Sciences achieved a 99.35% passing rate",                          # department forms
    "Is the School of Psychology planning a Day 19 activity?",
    "Students from the Technology and Engineering College attended the College Community Mass on August 5.",   # DYCI units
    "Junior High School and Grade School classes were suspended on September 10.",
    "SENIOR HIGH SCHOOL CLASSES ARE SUSPENDED", "Graduate School enrollment opens Monday",
    "Is the College Community Mass on August 12?",                                            # question word + generic
    "Dr. Yanga's Colleges suspended classes", "Dr. Yanga’s Colleges, Inc. welcomed delegates",   # DYCI names
    "The Dr. Yangas Colleges hosted", "Francisco Balagtas Academy was founded in 1950",
    "Ateneo Avenue is closed", "La Salle Street is flooded", "We ate at UP Town Center",       # places
    "Bocaue Private Schools Association meeting",                                            # plural keyword
    "Elida High School students have no classes", "Elida Campus High School recognition day",  # DYCI campuses
    "Marciano High School has an exam", "MARCIANO CAMPUS SENIOR HIGH SCHOOL", "DYCI Elida Senior High School",
    "Bocaue High School has no classes", "Bocaue National High School has no classes",          # DYCI's town: uncertain
    "DYCI High School has no classes tomorrow", "DYCI Elementary School is closed",              # DYCI basic education
    "Dr. Yanga's Colleges High School department", "The Elementary School will hold its recognition day",
    "Junior High School Department of DYCI", "SHS and JHS classes are suspended",
    "classes are up again", "sabi ng ama ko",
])
def test_false_positives(text):
    assert detect(text) == []


# ---- mentions and uncertainty --------------------------------------------------------------------------------------------
def test_distinctive_core_counts_as_mention():
    assert mentions("delegates from Mater Dei Academy and Torrens", "Mater Dei Academy")
    assert mentions("We visited Philippine Science HS", "PHILIPPINE SCIENCE HIGH SCHOOL")
    assert mentions("At the conference at Torrens University Australia", "Torrens University")
    assert not mentions("DYCI ADVISORY in-person classes suspended", "Philippine Science High School")
    assert not mentions("Congratulations to our DYCIan nurses, 99.35%", "Hogwarts Academy")


def test_a_short_one_word_core_is_uncertain_and_counts_as_mentioned():
    assert mentions("Anything at all", "Abc Academy")


def _result(verdict, text):
    post = NS(document_id="p", text=text)
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND", retrieved_evidence=[post],
                              used_evidence=post, classification_result=None, explanation="E.")


DYCI_N06 = "Congratulations to our DYCIan Registered Nurses and the College of Health Sciences for a 99.35% passing rate."
DYCI_SEPT10 = "DYCI ADVISORY 10 September 2026 IN-PERSON CLASSES at ALL LEVELS are SUSPENDED today."


# ---- C2 on top of detector v2 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("message, post", [
    ("Philippine Science High School suspended in-person classes on September 10 because of the Southwest Monsoon.", DYCI_SEPT10),
    ("Hogwarts Academy achieved a 99.35% passing rate among first-time takers in the August 2026 nursing licensure exam.", DYCI_N06),
])
def test_targets_copied_dyci_facts_are_withheld(message, post):
    out, info = osc.check(message, _result("SUPPORTED", post))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"]


@pytest.mark.parametrize("message, post", [
    ("DYCI suspended in-person classes on September 10 because of the Southwest Monsoon.", DYCI_SEPT10),              # K08
    ("The College of Health Sciences achieved a 99.35% passing rate among first-time takers.", DYCI_N06),             # K04
    ("The University will hold the midterm examinations on September 28 to 30.", "Midterms on September 28-30."),   # K01
    ("THE MIDTERM EXAMINATIONS ARE SCHEDULED ON SEPTEMBER 28 UP TO 30, 2026.", "Midterms on September 28-30."),       # K02
    ("Junior High School and Grade School classes were suspended on September 10.", DYCI_SEPT10),                     # K06
    ("DYCI researchers presented a study at Torrens University in Brisbane.", "conference at Torrens University Australia"),  # K07
])
def test_controls_and_must_keep_cases_are_untouched(message, post):
    original = _result("SUPPORTED", post)
    out, _ = osc.check(message, original)
    assert out is original


def test_several_institutions_one_missing_is_withheld_with_the_missing_one_listed():
    out, info = osc.check("Mater Dei Academy and Hogwarts Academy joined the fiesta",
                          _result("SUPPORTED", "Guests from Mater Dei Academy joined the fiesta."))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["missing_in_post"] == ["Hogwarts Academy"]


@pytest.mark.parametrize("message", ["BSU suspended classes on September 10",
                                     "CLASSES AT UP AND AMA ARE SUSPENDED ON SEPTEMBER 10, 2026"])
def test_uncertain_identification_never_changes_a_verdict(message):
    original = _result("SUPPORTED", DYCI_SEPT10)
    assert osc.check(message, original)[0] is original


# ---- C1 on top of detector v2 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("message", [
    "The University will hold the midterm examinations on September 28 to 30.",
    "MIDTERMS AT DYCI ARE ON SEPTEMBER 28 UP TO 30", "AMA KO SAID WALANG PASOK", "Ateneo Avenue is closed",
    "Junior High School and Grade School classes were suspended on September 10.",
])
def test_c1_release_review_probes_give_no_notice(message):
    assert notice(message, [NS(document_id="p", text=DYCI_SEPT10)]) is None


def test_dyci_campus_and_town_names_never_trigger_c2():
    for message in ("Elida High School classes were suspended on September 10.",
                    "Marciano High School classes were suspended on September 10.",
                    "Bocaue High School classes were suspended on September 10."):
        original = _result("SUPPORTED", DYCI_SEPT10)
        assert osc.check(message, original)[0] is original, message


def test_c1_targets_get_a_notice():
    assert notice("Hogwarts Academy achieved a 99.35% passing rate", [NS(document_id="n06", text=DYCI_N06)])["schools"] == ["Hogwarts Academy"]
    assert notice("Philippine Science High School suspended classes", [NS(document_id="s", text=DYCI_SEPT10)])["schools"] == \
        ["Philippine Science High School"]
