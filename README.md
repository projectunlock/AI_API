## pre-requisite

1. download docker
2. download model ```model-0000.params``` and put into folder ```mtcnninsightface/insightface-model/```. Link: https://drive.google.com/file/d/1Da1cnmQYSqhHzZH7ejVOoP9VfKTqHFML/view?usp=sharing, 
3. download cloth tag model ```mask_rcnn_Modanet.h5```  and put into folder ```modanet```. Link: https://drive.google.com/file/d/1narGkkyBS1TTtpWCPErsieAX7zozarde/view?usp=sharing


## build and run
```build docker image
docker-compose build (re-build everytime there is an update)
```

```run docker image
docker-compose up
```


## curl commands


```
fake: export url=http://ec2-34-247-255-170.eu-west-1.compute.amazonaws.com:5000
local real: export url=http://localhost:5000
```



1. Face_id
```
train:
curl ${url}/face_id -Fmethod=train -Fimage=@image.png  -Fuser_id="allie"  // this keeps upload, return train status
unlock:
curl ${url}/face_id -Fmethod=unlock -Fimage=@image.png // return user_id
status:
curl ${url}/face_id -Fmethod=status -Fuser_id="allie" // return status
```


<!-- 2. Virtual Background
```
curl ${url}/virtual_background -Fimage=@image.png  -Fbackground=@background.png // return virtual background image
curl ${url}/virtual_background -Fimage=@image.png  -Fbackground=@background.png > output.png   // to save the returned image
``` -->


3. Cloth Parsing
```
auto-tag:
curl ${url}/close_parsing -Fmethod=tag -Fimage=@image.png  -Fuser_id=allie // return a dictionary of {'bboxes': bboxes, "scores": scores, "labels": labels}
search:
curl ${url}/close_parsing -Fmethod=search -Ftag=top -Fuser_id=allie // return image files with the relevant keyword. Empty if not available
```


4. Health
```
curl ${url}/health
```


## params
to adjust the minimal number of images for face_id train, modify line 18 in app.py parameter ```min_train```