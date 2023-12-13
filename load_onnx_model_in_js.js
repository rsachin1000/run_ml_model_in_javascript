const onnx = require("onnxjs");
const fs = require("fs");
const { PNG } = require("pngjs");

// Load an ONNX model. This model takes a 1*3*32*32 image and classifies it.
function loadModel() {
  const session = new onnx.InferenceSession();
  console.log("Loading model...");
  const model = fs.readFileSync("./model.onnx");
  console.log("Creating tensor...");
  session.loadModel(new Uint8Array(model));
  console.log("Model loaded.");
  return session;
}

// Load image.
function loadImage(filePath) {
  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(new PNG())
      .on("parsed", function () {
        const numChannels = 3; // For RGB
        const normalizedTensor = new Float32Array(
          this.width * this.height * numChannels
        );

        for (let y = 0; y < this.height; y++) {
          for (let x = 0; x < this.width; x++) {
            let idx = (this.width * y + x) << 2;
            for (let channel = 0; channel < numChannels; channel++) {
              // Normalize each channel
              normalizedTensor[numChannels * (this.width * y + x) + channel] =
                (this.data[idx + channel] / 255 - 0.5) / 0.5;
            }
          }
        }
        resolve(
          new onnx.Tensor(normalizedTensor, "float32", [
            1,
            3,
            this.height,
            this.width,
          ])
        );
      })
      .on("error", (err) => reject(err));
  });
}

loadImage("./image_classification_model/sample_images/0_0.png")
  .then((imageTensor) => {
    console.log(imageTensor);
    let session = loadModel();
    // const outputMap = session.run([imageTensor]);
    // // const outputData = outputMap.values().next().value.data;
    // // const outputProbabilities = Array.from(outputData);
    // // const maxIndex = outputProbabilities.indexOf(Math.max(...outputProbabilities));
    // console.log(`Predicted class is ${outputMap}`);
  })
  .catch((error) => console.error(error));
