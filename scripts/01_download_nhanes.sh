#!/usr/bin/env bash
# Download NHANES 2005-2018 DEMO + DPQ (depression) files from CDC.
# CDC public data, current path pattern (post-2024 site redesign).
set -e
cd "$(dirname "$0")/../data/nhanes_raw"
declare -A CY=( [2005]=D [2007]=E [2009]=F [2011]=G [2013]=H [2015]=I [2017]=J )
for yr in 2005 2007 2009 2011 2013 2015 2017; do
  s=${CY[$yr]}
  for f in DEMO DPQ; do
    curl -sL -A "Mozilla/5.0 research" \
      -o "${f}_${s}.xpt" \
      "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/${yr}/DataFiles/${f}_${s}.xpt" \
      -w "%{http_code} %{size_download}  ${f}_${s}.xpt\n"
  done
done
