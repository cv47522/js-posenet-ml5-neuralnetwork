// ml5.js: Pose Classification
// Step 1: Data Collection: https://editor.p5js.org/codingtrain/sketches/kTM0Gm-1q
// Step 2: Model Training: https://editor.p5js.org/codingtrain/sketches/-Ywq20rM9
// Step 3: Model Deployment: https://editor.p5js.org/codingtrain/sketches/c5sDNr8eM
/* ===
Available parts in keypoints array are:
Index Part
0   nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle
=== */

let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "";

let state = 'waiting';
let targetLabel;

const dataFile = 'abcd.json';
const options = {
  // dataUrl: dataFile,
  inputs: 34, // 17 * 2
  outputs: 4,
  task: 'classification',
  debug: true
}

const modelInfo_ABCD = {
  model: 'model_01234/model.json',
  metadata: 'model_01234/model_meta.json',
  weights: 'model_01234/model.weights.bin',
};
const modelInfo_YMCA = {
  model: 'model_ymca/model.json',
  metadata: 'model_ymca/model_meta.json',
  weights: 'model_ymca/model.weights.bin',
};
//------------------------ UI ------------------
let saveDataBtn;
let loadFileBtn;
let trainModelBtn;
let loadModelInput, loadModelBtn_YMCA, loadModelBtn_ABCD;
// let info;
let message;
//----------------------------------------------

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
//------------------------ Classifiers ------------------
   poseNet = ml5.poseNet(video, posenetModelLoaded);
   poseNet.on('pose', gotPoses);
   brain = ml5.neuralNetwork(options);
//-------------------------------------------------------
   // LOAD PRETRAINED MODEL
   // brain.load(modelInfo, brainLoaded);
   //  LOAD TRAINING DATA
   // brain.loadData(dataFile, trainModel);
//------------------------ UI ------------------
  saveDataBtn = select('#saveDataBtn');
  saveDataBtn.mousePressed(saveJsonData);

  loadFileBtn = select('#loadFileBtn');
  // loadFileBtn.mousePressed(() => {
  //  brain.loadData(dataFile, trainModel);
  // }));
  loadFileBtn.changed(() => {
    brain.loadData(loadFileBtn.elt.files, () => {
      console.log('Data Loaded!');
      message.html('Data Loaded!');
      console.log(loadFileBtn.elt.files);
    });
  });

  trainModelBtn = select('#trainModelBtn');
  trainModelBtn.mousePressed(trainModel);


  loadModelBtn_YMCA = select('#loadModelBtn_YMCA');
  loadModelBtn_YMCA.mousePressed(() => {
    // brain.load('model/ymca_model.json', brainLoaded);
    brain.load(modelInfo_YMCA,  () => {
      brainLoaded();
      message.html('<br> YMCA Model Loaded!', true);
    });
  });

  loadModelBtn_ABCD = select('#loadModelBtn_ABCD');
  loadModelBtn_ABCD.mousePressed(() => {
    // brain.load('model/01234_model.json', brainLoaded);
    brain.load(modelInfo_ABCD,  () => {
      brainLoaded();
      message.html('<br> 01234 Model Loaded!', true);
    });
  });

  loadModelInput = select('#loadModelInput');
  loadModelInput.changed(() => {
    brain.load(loadModelInput.elt.files, () => {
      brainLoaded();
      message.html('<br> External Model Loaded!', true);
    });
    console.log(loadModelInput.elt.files);
  });

  message = select('#message');
//----------------------------------------------

}




function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);
  // filter(INVERT);
  noStroke();
  fill(255);
  video.loadPixels();
  const stepSize = 10;
  for (let y = 0; y < height; y += stepSize) {
    for (let x = 0; x < width; x += stepSize) {
      const i = y * width + x;
      const darkness = (255 - video.pixels[i * 4]) / 255;
      const radius = stepSize * darkness;
      ellipse(x, y, radius, radius);
    }
  }

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(4);
      stroke(255, 0, 0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0, 255, 0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  pop();

  fill(255, 0, 255);
  noStroke();
  textSize(512);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
}

//------------------------ Customized Functions ------------------
function posenetModelLoaded() {
  console.log('poseNet ready');
  message.html('poseNet ready');
}

function externalModelLoaded() {
  console.log('External Model Loaded');
  message.html('External Model Loaded');
}

function brainLoaded() {
  console.log('Pose Classification Ready!');
  message.html('Pose Classification Ready!');
  classifyPose();
}

function trainModel() {
  brain.normalizeData();
  brain.train({
    epochs: 50
  }, whileTraining, finishedTraining);
  console.log('Data Ready! Training Model...');
  message.html('Data Ready! Training Model...');
}

function whileTraining(epoch, loss) {
  message.html('Epoch Now: ' + epoch);
}

function finishedTraining() {
  console.log('Model trained!');
  message.html('Model trained!');
  brain.save(); // Save Model
  classifyPose();
}

function saveJsonData() {
  brain.saveData();
}

//------------------------
function gotPoses(poses) {
  // console.log(poses);
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}
//------------------------
function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {
  if (results[0].confidence > 0.75) {
    console.log(results);
    poseLabel = results[0].label.toUpperCase();
  }else if(error){
    console.log(error);
    return;
  }
  classifyPose();
}
//------------------------
function keyPressed() {
  if (key == 't') {
    message.html('Start training...');
    brain.normalizeData();
    brain.train({epochs: 50}, whileTraining, finishedTraining);
  } else if (key == 's') {
    brain.saveData();
  } else {
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(() => {
      console.log('collecting');
      message.html('Collecting ' + targetLabel.toUpperCase() + ' Pose Data...');
      state = 'collecting';
      setTimeout(() => {
        console.log('not collecting');
        message.html('Not collecting...');
        state = 'waiting';
      }, 2000);
    }, 1000);
  }
}
