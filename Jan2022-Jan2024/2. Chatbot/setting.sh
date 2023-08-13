source set_env.sh

if [ $# -ne 0 ]; then
  print_message ERROR "Process failed. No parameters are needed."
  exit 1
fi

# # 1 day after
# START_DATE=`date '+%Y%m%d' -d "\`date '+%Y%m%d' -d ${ARG_DATE}\` 1 days"`
# # 3 days after
# END_DATE=`date '+%Y%m%d' -d "\`date '+%Y%m%d' -d ${ARG_DATE}\` 3 days"`
date = 20230101
if [ -z ${TARGET_DATE} ]; then
  TARGET_DATE=`date '+%Y%m%d' -d "+1 day"`
  print_message INFO "TARGET_DATE is not set, defaulting to current date + 1"
fi

if [ -z ${EXT_DATE} ]; then
  EXT_DATE = `date '+%Y%m%d'`
  print_message INFO "extract_date is not set, defaulting to current date"
fi

printenv | grep PRED_BY_MODE
printenv | grep EXT_DATE
printenv | grep TARGET_DATE

year=`echo ${TARGET_DATE} | cut -c1-4`
month=`echo ${TARGET_DATE} | cut -c5-6`

LOCAL_PATH=`echo /mnt/work/tmp/`

# for python env
# export PYTHONPATH=/usr/local/lib/python3.7/site-packages:/usr/local/lib64/python3.7/site-packages:${PROJECT_HOME}/global:${PROJECT_HOME}/lib:${PYTHONPATH}
while [ 1 ] ; do
    # Argument by PRED_BY_MODE
    print_message INFO "Run main prediction program. Extract date is ${EXT_DATE}, predict date is ${TARGET_DATE}."
    # python3 ${PROJECT_HOME}/app/option1.py ${EXT_DATE} ${TARGET_DATE}
    print_message "Option 1"

    if [ $? -ne 0 ];
    then
        print_message ERROR "Process failed. line: ${LINENO} on ${0}"
        exit 1
    fi
    # python3 ${PROJECT_HOME}/app/option2.py ${EXT_DATE} ${TARGET_DATE}
    print_message "Option 2"

    if [ $? -ne 0 ];
    then
        print_message ERROR "Process failed. line: ${LINENO} on ${0}"
        exit 1
    fi

    if [ $TARGET_DATE = $END_DATE ] ; then
        break
    fi

    TARGET_DATE=`date -d "$TARGET_DATE 1day" "+%Y%m%d"`

done
exit 0