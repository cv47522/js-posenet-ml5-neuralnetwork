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
let maxDistance;

let system;

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
//------------------------ UI & Sound ------------------
let muteBtn;
let unMuteBtn;
let saveDataBtn;
let loadFileBtn;
let trainModelBtn;
let loadModelInput, loadModelBtn_YMCA, loadModelBtn_ABCD;
// let info;
let message;
let label;

let virus_sound;
//----------------------------------------------

function setup() {
  createCanvas(640, 480); // NO WEBGL!
  video = createCapture(VIDEO);
  video.hide();

  // virus_sound = loadSound('creature.wav');
  virus_sound = loadSound('scifi.wav');

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
  muteBtn = select('#muteBtn');
  muteBtn.mousePressed(() => {
    virus_sound.volume(0);
    virus_sound.stop();
  });

  unMuteBtn = select('#unMuteBtn');
  unMuteBtn.mousePressed(() => {
    virus_sound.volume(1);
    // let fs = fullscreen();
    // fullscreen(!fs);
  });

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
      message.html('<br> YLIE Model Loaded!', true);
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
  // noStroke();
  // fill(255);
  // video.loadPixels();
  // const stepSize = 10;
  // for (let y = 0; y < height; y += stepSize) {
  //   for (let x = 0; x < width; x += stepSize) {
  //     const i = y * width + x;
  //     const darkness = (255 - video.pixels[i * 4]) / 255;
  //     const radius = stepSize * darkness;
  //     ellipse(x, y, radius, radius);
  //   }
  // }

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0, 0, 255);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0, 0, 255);
      stroke(0);
      ellipse(x, y, 16, 16);
    }

  }
  pop();


  switch (poseLabel) {
    case 'Y':
      let shoulderR = pose.rightShoulder;
      let shoulderL = pose.leftShoulder;
      let distance = dist(shoulderR.x, shoulderR.y, shoulderL.x, shoulderL.y);

      let wristR = pose.rightWrist;
      let wristL = pose.leftWrist;

      let x = map((wristR.x + wristL.x) / 2, 0, width, width, 0);
      let y = (wristR.y + wristL.y) / 2;
      maxDistance = sqrt(pow(width, 2) + pow(height, 2));
      createVirus(x, y, map(distance, 0, maxDistance, maxDistance,0));
      // createVirus(x, y, distance);

      // Set the rate to a range between 0.1 and 4
      // Changing the rate alters the pitch
      let speed = map(distance, 0.1, maxDistance, 0.1, 4);
      // speed = constrain(speed, 0.01, 4);
      virus_sound.rate(speed * 3);
      virus_sound.play();
      // console.log('Vrius X: ' + x + ', ' + 'Wrist R X: ' + wristR.x);
      break;
    case 'B':
      label = 'STOP!';
      createText(label);
      // let wristL = pose.leftWrist;
      // system = new ParticleSystem(createVector(wristL.x, wristL.y));
      // system.addParticle();
      // system.run();
      console.log(label);
      break;
    case 'M':
      label = '?';
      createText(label);
      // console.log(label);
      break;
    default:
      virus_sound.stop();
      // system.stop();
      createText(poseLabel);
  }
}

//------------------------ Customized Functions ------------------
function star(x, y, radius1, radius2, npoints) {
  let angle = TWO_PI / npoints;
  let halfAngle = angle / 2.0;
  beginShape();
  for (let a = 0; a < TWO_PI; a += angle) {
    let sx = x + cos(a) * radius2;
    let sy = y + sin(a) * radius2;
    vertex(sx, sy);
    sx = x + cos(a + halfAngle) * radius1;
    sy = y + sin(a + halfAngle) * radius1;
    vertex(sx, sy);
  }
  endShape(CLOSE);
}

function createVirus(x, y, distance) {
  push();
  // scale(-1, 1);
  stroke(0);
  strokeWeight(2);
  if (distance >= 710){
    fill(0, 255, 0)
  }else if (distance >= 650 && distance < 710){
    fill(255, 255, 0)
  }else if (distance >= 600 && distance < 650){
    fill(255, 125, 0)
  }else{
    fill(255, 0, 0)
  }
  // console.log(distance);

  translate(x, y);
  rotate(frameCount / 50.0);
  star(0, 0, 0.1 * distance, 0.15 * distance, 40);
  pop();
}

function createText(labelShow) {
  fill(255, 0, 255);
  noStroke();
  textSize(200);
  textAlign(CENTER, CENTER);
  text(labelShow, width / 2, height / 2);
  // text(label, width / 2, height / 2);
}
//------------------------ Particle Functions ------------------
// A simple Particle class
// let Particle = function(position) {
//   this.acceleration = createVector(0, 0.05);
//   this.velocity = createVector(random(-1, 1), random(-1, 0));
//   this.position = position.copy();
//   this.lifespan = 255;
// };
//
// Particle.prototype.run = function() {
//   this.update();
//   this.display();
// };
//
// // Method to update position
// Particle.prototype.update = function(){
//   this.velocity.add(this.acceleration);
//   this.position.add(this.velocity);
//   this.lifespan -= 2;
// };
//
// // Method to display
// Particle.prototype.display = function() {
//   stroke(200, this.lifespan);
//   strokeWeight(2);
//   fill(127, this.lifespan, 255);
//   ellipse(this.position.x, this.position.y, 12, 12);
// };
//
// // Is the particle still useful?
// Particle.prototype.isDead = function(){
//   return this.lifespan < 0;
// };
//
// let ParticleSystem = function(position) {
//   this.origin = position.copy();
//   this.particles = [];
// };
//
// ParticleSystem.prototype.addParticle = function() {
//   this.particles.push(new Particle(this.origin));
// };
//
// ParticleSystem.prototype.run = function() {
//   for (let i = this.particles.length-1; i >= 0; i--) {
//     let p = this.particles[i];
//     p.run();
//     if (p.isDead()) {
//       this.particles.splice(i, 1);
//     }
//   }
// };
//--------------------------------------------------------------------
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
    epochs: 80
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
    // console.log(results);
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
    brain.train({epochs: 80}, whileTraining, finishedTraining);
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
      }, 5000);
    }, 1000);
  }
}
