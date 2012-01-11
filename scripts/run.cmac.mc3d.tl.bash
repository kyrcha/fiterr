#!/bin/bash
clear
for ((  i = 0 ;  i <= 10;  i++  ))
do
	echo $i
	java -cp "../bin/:../lib/commons-math-2.0.jar:../lib/Jama-1.0.2.jar:../lib/JavaRLGlueCodec.jar:../lib/log4j-1.2.15.jar" info.kyrcha.fiterr.rlglue.Platform ../params/rl-glue/experiments/experiment.cmac.mc3d.tl.params >> mc3d.tl1.1.out
done

