async function loadModel() {
  const session = new onnx.InferenceSession();
  console.log("Session created");
  await session.loadModel("http://localhost:8000/model.onnx");
  console.log("Model loaded");
  return session;
}

async function preprocessImage(image) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);

  let imageData = ctx.getImageData(0, 0, image.width, image.height);

  let data = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    data.push((imageData.data[i] / 255 - 0.5) / 0.5);
    data.push((imageData.data[i + 1] / 255 - 0.5) / 0.5);
    data.push((imageData.data[i + 2] / 255 - 0.5) / 0.5);
  }

  return new onnx.Tensor(data, "float32", [1, 3, image.height, image.width]);
}

async function objectClassName(array) {
  const names = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
  ];
  console.log("Array: ", names);
  const idxOfMaxValue = array.reduce(
    (maxIndex, value, index, arr) => (value > arr[maxIndex] ? index : maxIndex),
    0
  );
  let name = names[idxOfMaxValue];
  return name;
}

document
  .getElementById("imageUpload")
  .addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = async () => {
      const tensor = await preprocessImage(image);
      console.log("Image preprocessed");

      const session = await loadModel();
      console.log("Session loaded");

      const outputMap = await session.run([tensor]);
      console.log("Session run");
      const outputData = outputMap.values().next().value.data;
      console.log("Output data");
      console.log(outputData);

      const className = await objectClassName(outputData);
      console.log("Class of Identified Object: ", className);

      const displayElement = document.getElementById("classNameDisplay");
      displayElement.textContent = "Identified Object: " + className;
      displayElement.style.display = "block";
    };
  });
