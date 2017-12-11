#!/bin/sh
#US stand for UnSatisfied
#S stand for Statisfied
for i in "$@"
do
   :
   # do whatever on $i
done
nbPigeonS=3;
nbPigeonnierS=3;
nbPigeonUS=6;
nbPigeonnierUS=5;


{
   SgpuTime=$(./bin/gpu_satisfy.exe $nbPigeonS $nbPigeonnierS | grep "gpu time")
   ScpuTime=$(./bin/satisfy.exe $nbPigeonS $nbPigeonnierS | grep "cpu time")
}  || {
    echo "the execution of the program failed please check your setup"
}
{
   USgpuTime=$(./bin/gpu_satisfy.exe $nbPigeonS $nbPigeonnierS | grep "gpu time")
   UScpuTime=$(./bin/satisfy.exe $nbPigeonUS $nbPigeonnierUS | grep "cpu time")
}  || {
    echo "the execution of the program failed please check your setup"
}
echo "satisfying instance"
echo $SgpuTime
echo $ScpuTime
echo "unsatisfying instance"
echo $USgpuTime
echo $UScpuTime
echo "First arg: $1"
echo "Second arg: $2"
echo "List of all arg: $@"
