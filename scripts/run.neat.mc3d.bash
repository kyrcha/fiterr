#!/bin/sh
clear
java -cp "../bin/:../lib/commons-math-2.0.jar:../lib/Jama-1.0.2.jar:../lib/JavaRLGlueCodec.jar:../lib/log4j-1.2.15.jar" info.kyrcha.fiterr.rlglue.Platform ../params/rl-glue/experiments/experiment.neat.mc3d.params
