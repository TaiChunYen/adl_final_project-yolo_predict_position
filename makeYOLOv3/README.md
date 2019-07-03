## train YOLOv3
python train.py  
(detail in https://chtseng.wordpress.com/2018/10/08/%E5%A6%82%E4%BD%95%E5%BF%AB%E9%80%9F%E5%AE%8C%E6%88%90yolo-v3%E8%A8%93%E7%B7%B4%E8%88%87%E9%A0%90%E6%B8%AC/)

## test YOLOv3
python playYOLO.py -i [picture name]  
python playYOLO.py -v [video name]  

## print yolo roi position in test.json
python printjson.py -i [picture name](for one picture)  
bash p_json.sh [file/dir](for many picture)  
(test.json must have [] before run printjson.py or p_json.sh)

test.json  
--'output'=[id,acr,left,top,left + width,top + height] 

## reference
https://github.com/ch-tseng/makeYOLOv3


