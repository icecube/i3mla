mkdir -p ./data/

curl --insecure https://data.hawc-observatory.org/datasets/crab_data/public_data/crab_2017/HAWC_9bin_507days_crab_response.hd5 -o data/HAWC_9bin_507days_crab_response.hd5
curl --insecure https://data.hawc-observatory.org/datasets/crab_data/public_data/crab_2017/HAWC_9bin_507days_crab_data.hd5 -o data/HAWC_9bin_507days_crab_data.hd5

#MD5 (HAWC_9bin_507days_crab_data.hd5) = fcccc288bdef0f10dad49b856cfc9ffd
#MD5 (HAWC_9bin_507days_crab_response.hd5) = 4dd2e333a5470f55c4b34d919c277142