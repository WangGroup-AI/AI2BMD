import os

DEBUG_RC = "DEBUG_RUN_COMMAND" in os.environ
MAX_CYC  = os.environ.get("MAX_CYC")

if MAX_CYC:
    MAX_CYC = int(MAX_CYC)
