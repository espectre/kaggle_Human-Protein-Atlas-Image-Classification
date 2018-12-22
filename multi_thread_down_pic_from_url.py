#! /bin/bash
source /etc/profile;

# -----------------------------

tempfifo=$$.fifo        # $$表示当前执行文件的PID

# -----------------------------

trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo $tempfifo
exec 1000<>$tempfifo
rm -rf $tempfifo

for ((i=1; i<=100; i++))
do
    echo >&1000
done

cat url.txt|while read line
do
    read -u1000
    {
        wget $line
        echo >&1000
    } &

done

wait
echo "done!!!!!!!!!!"