# Author: sunhy,2023/3/12
# Copyright (c) 2023 APOCO Corporation
# All rights reserved.

#ps -ef | grep nameko | grep -v grep | awk '{print $(NF-2)}' | while read -r service_name; do
#    echo "$service_name"
#done | tee /dev/tty | wc -l

services=$(ps -ef | grep nameko | grep run | grep -v grep | awk '{print $(NF-2)}')

if [ -n "$services" ]; then
    echo "------------------------------------------"
    echo "The following Nameko services are running:"
    echo "------------------------------------------"
    echo "$services" | while read -r service_name; do
        echo "- $service_name"
    done

    count=$(echo "$services" | wc -l)
    echo "------------------------------------------"
    echo "Total: $count"
else
    echo "No Nameko services are running."
fi

