ps aux|grep 'python searchEnginePython' |grep -v grep  |awk '{print "kill -9 "$2}' |sh
