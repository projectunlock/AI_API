## to run

(pre-requisite, 1, download docker, 2 download model ```model-0000.params``` and put into folder ```mtcnninsightface/insightface-model/```)

```build docker image
docker-compose build
```

```run docker image
docker-compose up
```


## curl commands


```fake
fake: export url=http://ec2-34-247-255-170.eu-west-1.compute.amazonaws.com:5000
```

```local
local export url=http://localhost:5000
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
```


3. Cloth Parsing
```
curl ${url}/close_parsing -Fimage=@image.png   // return virtual background image
```
 -->

4. Health
```
curl ${url}/health
```


## params
to adjust the minimal number of images for face_id train, modify line 18 in app.py parameter ```min_train```