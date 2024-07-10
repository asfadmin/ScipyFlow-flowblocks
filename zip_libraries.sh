## Deploys flowblocks to the $1 directory, call like
## bash deploy_flowblocks.sh directory

#list all flowblock libraries into library_list.txt
ls -l ./flowblock_source_files/ | grep '^d' | awk '{$1=$2=$3=$4=$5=$6=$7=$8=""; print substr($0, 9)}' > ./flowblock_source_files/library_list.txt

#delete all zipped libraries
cat ./flowblock_source_files/library_list.txt | xargs -I {} bash -c 'rm ./flowblock_source_files/{}/*.zip'

#zip all library contents
cat ./flowblock_source_files/library_list.txt | xargs -I {} bash -c 'zip -j flowblock_source_files/{}/library.zip ./flowblock_source_files/{}/*.py'
