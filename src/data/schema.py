"""
Defines the shape of a valid campus claim/evidence record.

Keeping this in one small file means the "rules" for what counts as valid
data live in exactly one place. Every other file (loader, validation,
tests) imports from here instead of re-typing label lists or field names.
"""

# The three possible relationships between a claim and its evidence.
# Any label outside this set is invalid.
VALID_LABELS = {
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
}

# Where a piece of evidence originated. Kept as a CLOSED, enforced set
# (rather than free text) starting in Milestone 5, specifically so that
# future ingestion components -- website scraping, Facebook Graph API,
# manual entry -- cannot each invent their own slightly different label
# for the same concept (e.g. "official_site" vs "official-site" vs
# "university_website"). Retrieval and classification do not read this
# field today, but future evidence-weighting or filtering logic will
# depend on it being consistent.
#
# "synthetic" remains here deliberately -- it is NOT a future real
# source, it is how every dev/test fixture in this project is labeled,
# and removing it would break Milestones 1-4's existing dev dataset.
VALID_SOURCE_TYPES = {
    "synthetic",        # development/test data only -- never real thesis data
    "official_site",    # university website announcements/pages
    "facebook",         # official/recognized Facebook pages, via Graph API only
    "manual_entry",      # hand-transcribed/curated by the research team
}

# How much a given evidence source should be trusted, independent of how
# similar it looks to a claim (that's retrieval's job) or what relationship
# a classifier finds (that's classification's job). This field is OPTIONAL
# and currently UNUSED by retrieval/classification/verification -- it
# exists now so that when Facebook ingestion is built (a future milestone),
# there is already a place to record "this is an official page" vs. "this
# is an unverified student page" without changing the schema at that time.
# Until a future milestone actually implements credibility-weighted
# retrieval or verification, this field is descriptive metadata only.
VALID_CREDIBILITY_TIERS = {
    "official",     # university offices, registrar, official departments
    "recognized",   # recognized student orgs / official-but-secondary pages
    "unverified",   # anything not confirmed as an official/recognized source
}

# Fields that must be present in every row and must not be empty.
# Note: "evidence" is NOT in this list -- it's allowed to be an empty
# string (that's how INSUFFICIENT_EVIDENCE rows are represented), but the
# column itself must exist. See REQUIRED_COLUMNS below.
REQUIRED_NON_EMPTY_FIELDS = {
    "claim",
    "label",
    "source_type",
    "group_id",
}

# All columns that must exist in the CSV header, regardless of whether
# their value is allowed to be empty.
#
# "credibility_tier" is included as a REQUIRED COLUMN (must exist) but its
# value MAY be empty (not a REQUIRED_NON_EMPTY_FIELD) -- existing
# synthetic dev rows and any future source that hasn't been credibility-
# reviewed yet can leave it blank without failing validation.
REQUIRED_COLUMNS = {
    "claim",
    "evidence",
    "label",
    "source_type",
    "source_url",
    "date",
    "group_id",
    "credibility_tier",
}
