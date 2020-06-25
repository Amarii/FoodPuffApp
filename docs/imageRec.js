
const URL = "https://teachablemachine.withgoogle.com/models/wcWYT42HY/";

let model, webcam, labelContainer, maxPredictions;
let index = 0;

// Load the image model and setup the webcam
async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    // load the model and metadata
    // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
    // or files from your local hard drive
    // Note: the pose library adds "tmImage" object to your window (window.tmImage)
    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Convenience function to setup a webcam
    const flip = true; // whether to flip the webcam
    webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
    await webcam.setup(); // request access to the webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // append elements to the DOM
    document.getElementById("webcam-container").appendChild(webcam.canvas);
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) { // and class labels
        labelContainer.appendChild(document.createElement("img"));
    }
    let index;
}
let msg = '';

async function loop() {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

// run the webcam image through the image model
async function predict() {
    // predict can take in an image, video or canvas html element
    const prediction = await model.predict(webcam.canvas);
    const highestChance = Math.max(prediction[0].probability.toFixed(2), prediction[1].probability.toFixed(2), prediction[2].probability.toFixed(2), prediction[3].probability.toFixed(2))

    for (let i = 0; i < maxPredictions; i++) {
        if (highestChance == prediction[i].probability.toFixed(2)) {
            labelContainer.childNodes[0].style.width = "500px"
            labelContainer.childNodes[0].style.height = "300px"
            labelContainer.childNodes[0].innerHTML = prediction[i].className + "!"
            labelContainer.childNodes[0].alt = prediction[i].className
            msg.text = prediction[i].className


            // if (index == 50) {
            //     //window.speechSynthesis.speak(msg)
            //     index = 0;
            // }
            //index++
        }
    }

}

    
    
