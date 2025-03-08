// ! Functions that deal with button events

// * Preview switch
$(function () {
    $("a#cam-preview").bind("click", function () {
      $.getJSON("/request_preview_switch", function (data) {
        // do nothing
      });
      return false;
    });
  });
  
  // * Model switch
  $(function () {
    $("a#use-model").bind("click", function () {
      $.getJSON("/request_model_switch", function (data) {
        // do nothing
      });
      return false;
    });
  });
  

  
  // * Camera Toggle (to blur/unblur the video feed)
  document.getElementById('cameraToggle').addEventListener('click', function () {
    const bg = document.getElementById('bg');
    const isBlurred = bg.classList.contains('blurred');
  
    if (isBlurred) {
      bg.classList.remove('blurred'); // Remove blur (camera on)
      this.textContent = "Disable Camera";
    } else {
      bg.classList.add('blurred'); // Add blur (camera off)
      this.textContent = "Enable Camera";
    }
  });
  
  
  document.getElementById('modelToggle').addEventListener('click', function () {
    fetch("/request_model_switch")
      .then(response => {
        if (!response.ok) {
          console.error("Failed to toggle model.");
          return;
        }
        this.textContent = this.textContent === "Disable Detection"
          ? "Enable Detection"
          : "Disable Detection";
      })
      .catch(error => console.error("Error:", error));
  });
  
  
  