#! /sh
function print_message () {
    local _level="$1"
    local _message="$2"

    printf "%s %s %s __shell__ %s\n" "`date '+%Y-%m-%d'`" "`date '+%H:%M:%S,%3N'`" "$_level" "$_message"
}

export PROJECT_HOME=pwd
if [[ -z ${PROJECT_HOME} ]]; then
    echo "ERROR env parameter PROJECT_HOME can not be empty."
    exit 1
fi

export RUN_ENV=master
if [[ -z ${RUN_ENV} ]]; then
    print_message ERROR "env parameter RUN_ENV can not be empty."
    exit 1
fi

export PRED_BY_MODE=daily
if [[ -z ${PRED_BY_MODE} ]]; then
    print_message ERROR "env parameter PRED_BY_MODE can not be empty and must be daily or monthly."
    exit 1
fi

if [[ $PRED_BY_MODE == 'daily' ]]; then
    echo "you have selected PRED_BY_MODE: $PRED_BY_MODE"
    # apply logic of prediction of daily
    # clone repo contains daily

elif [[ $PRED_BY_MODE == 'monthly' ]]; then
    # apply logic of prediction of monthly
    echo "You have selected PRED_BY_MODE $PRED_BY_MODE"
    
else
    echo "You must select 1 for monthly and 2 for daily, any other value is invalid!!"
    break
fi