#!/bin/bash

# usage: sh get_log.sh out.beh2-ph512xx4

log_file=$1
grep Begin $log_file -A 200 | grep End -B 200| sed '/Begin/d' | sed '/End/d'
