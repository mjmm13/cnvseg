#!/usr/bin/env python
from math import sqrt
import sys

from genomedata import Genome
from numpy import array, arcsinh
from path import Path
from segway import run
from segway.input_master import DTParamSpec
from segway.gmtk.input_master import (Covar, DenseCPT, DeterministicCPT,
                                      DPMF, InlineSection,
                                      InputMaster, Mean, NameCollection, 
                                      Object)

GENOMEDATA_FILE = "genomedata"
TRAINDIR = "segway_output/traindir"
ANNOTATEDIR = "segway_output/annotatedir"

### Set up Directories and run train-init to generate all important files ###
genomedata = Path(GENOMEDATA_FILE)
traindir = Path(TRAINDIR)
annotatedir = Path(ANNOTATEDIR)

run.main(["train-init", "--num-labels=3", "--resolution=50", genomedata, traindir])

### Create InputMaster object and begin adding sections, starting with DT ###
class DTVariables(Object):
    num_segs = 3
    seg_countdowns_initial = [1, 1, 1]
    supervision_type = 0

DTObject = DTVariables
DT = DTParamSpec(DTObject)

input_master = InputMaster()

### Name Collection ###
input_master.append(InlineSection(("collection_seg_LogR", NameCollection(["mx_seg0_subseg0_LogR",
                                                                           "mx_seg1_subseg0_LogR",
                                                                           "mx_seg2_subseg0_LogR"]))))

### Deterministic CPT ### 
input_master.append(InlineSection(("seg_segCountDown",
                                    DeterministicCPT([1, "CARD_SEG", "CARD_SEGCOUNTDOWN",
                                                      "map_seg_segCountDown"])),
                                   ("frameIndex_ruler",
                                    DeterministicCPT([1, "CARD_FRAMEINDEX", "CARD_RULER",
                                                      "map_frameIndex_ruler"])),
                                   ("segTransition_ruler_seg_segCountDown_segCountDown",
                                    DeterministicCPT([4, "CARD_SEGTRANSITION", "CARD_RULER",
                                                      "CARD_SEG", "CARD_SEGCOUNTDOWN",
                                                      "CARD_SEGCOUNTDOWN", "map_segTransition_ruler_seg_segCountDown_segCountDown"])),
                                   ("seg_seg_copy",
                                    DeterministicCPT([1, "CARD_SEG", "CARD_SEG",
                                                      "internal:copyParent"])),
                                   ("subseg_subseg_copy",
                                    DeterministicCPT([1, "CARD_SUBSEG", "CARD_SUBSEG",
                                                      "internal:copyParent"]))))

### DenseCPT ###
input_master.append(InlineSection(("start_seg", DenseCPT([1/3, 1/3, 1/3])),
                                  ("seg_subseg", DenseCPT([[1.0], [1.0], [1.0]])),
                                  ("seg_seg", 
                                   DenseCPT([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])),
                                  ("seg_subseg_subseg",
                                   DenseCPT([[[1.0]], [[1.0]], [[1.0]]])),
                                  ("segCountDown_seg_segTransition",
                                   DenseCPT([[[0.99, 0.00999, 0.00001],
                                   [0.99, 0.00999, 0.00001],
                                   [0.99, 0.00999, 0.00001]],
                                   [[0.99, 0.01, 0.0],
                                   [0.99, 0.01, 0.0],
                                   [0.99, 0.01, 0.0]]]))))

### Mean and Covar Sections ###
# These sections are taken from the genomedata archive, and what we aim to change
with Genome(genomedata) as genome:
    # Load info from GD archive to get mean and variance
    sums = genome.sums
    sums_squares = genome.sums_squares
    num_datapoints = genome.num_datapoints

mean = sums / num_datapoints
var = (sums_squares / num_datapoints) - mean ** 2

sd = sqrt(var)
# Set group means to be 2 SD from the actual mean
means = [mean - 2 * sd, mean, mean + 2 * sd]
# Transform for arcsinh dist
var_transformed = arcsinh(var)
means_transformed = arcsinh(means)

input_master.append(InlineSection(("mean_seg0_subseg0_LogR", Mean(means_transformed[0])),
                                  ("mean_seg1_subseg0_LogR", Mean(means_transformed[1])),
                                  ("mean_seg2_subseg0_LogR", Mean(means_transformed[2]))))
input_master.append(InlineSection(("covar_LogR", Covar(var_transformed))))

### DPMF ###
input_master.append(InlineSection(("dpmf_always", DPMF([1.0]))))

### MC ###
input_master.append(InlineSection(("1 COMPONENT_TYPE_DIAG_GAUSSIAN mc_asinh_norm_seg0_subseg0_LogR", Object("mean_seg0_subseg0_LogR covar_LogR", "MC")),
                                  ("1 COMPONENT_TYPE_DIAG_GAUSSIAN mc_asinh_norm_seg1_subseg0_LogR", Object("mean_seg1_subseg0_LogR covar_LogR", "MC")),
                                  ("1 COMPONENT_TYPE_DIAG_GAUSSIAN mc_asinh_norm_seg2_subseg0_LogR", Object("mean_seg2_subseg0_LogR covar_LogR", "MC"))))

### MX ###
input_master.append(InlineSection(("1 mx_seg0_subseg0_LogR", Object("1 dpmf_always mc_asinh_norm_seg0_subseg0_LogR", "MX")),
                                   ("1 mx_seg1_subseg0_LogR", Object("1 dpmf_always mc_asinh_norm_seg1_subseg0_LogR", "MX")),
                                   ("1 mx_seg2_subseg0_LogR", Object("1 dpmf_always mc_asinh_norm_seg2_subseg0_LogR", "MX"))))

input_master_path = traindir / "params" / "input.master"

### Write out to input master ###
with open(input_master_path, "w") as filename:
    print("#include ", '"', traindir, '/auxiliary/segway.inc"\n\n', sep="",
          file=filename)
    print(DT, file=filename)
    print(input_master, file=filename)

run.main(["annotate", genomedata, traindir, annotatedir])
