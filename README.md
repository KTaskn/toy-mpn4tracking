# generates a datasets
``` shell
for i in `seq -f '%03g' 0 9`
do
    python gendata.py > datasets/$i.csv
done
```