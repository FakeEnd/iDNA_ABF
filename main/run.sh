array6=(6mAA 6mACEL 6mACEQ 6mAD 6mAF 6mAH 6mAR 6mAS 6mATT 6mATO 6mAX)
array5=(5hmCH 5hmCM)
array4=(4mCC 4mCF 4mCS 4mCT)

for((i=2;i<=2;i++));
do
  for((j=3;j<=3;j++));
  do
        nohup python -u train.py '-train-name' ${array4[i]}  '-test-name' ${array4[j]} '-device' 2 >"${array4[i]}_${array4[j]}.log" 2>&1 &
  done
done