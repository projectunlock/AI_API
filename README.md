## curl commands


```
export url=http://ec2-34-247-255-170.eu-west-1.compute.amazonaws.com:5000
```


1. Face_id
```
train:
curl ${url}/face_id -Fmethod=train -Fimage=@image.png  -Fuser_id="allie"  // this keeps upload, return train status
unlock:
curl ${url}/face_id -Fmethod=unlock -Fimage=@image.png // return user_id
```


2. Virtual Background
```
curl ${url}/virtual_background -Fimage=@image.png  -Fbackground=@background.png // return virtual background image
curl ${url}/virtual_background -Fimage=@image.png  -Fbackground=@background.png > output.png   // to save the returned image
```


3. Cloth Parsing
```
curl ${url}/close_parsing -Fimage=@image.png   // return virtual background image
```


4. Health
```
curl ${url}/Health

resp: 
{"version": 1, "healthy": true}
```
