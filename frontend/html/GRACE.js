/*
 * Transkit Online API
 * Testing JS file
 * (c) 2021 Vicomtech
 */


// Init
setStatus(0);


/*
 * Reset
 */

function doReset() {
  $("#masterkey").val("");
  $("#op_time").val("--");
  $("#transcription").val("");
  setStatus(0);
}

function emptyResults() {
  $("#transcription").val(" ");
  $("#op_time").text("--");
}


/*
 * Set translation status
 */

function setStatus(num) {
  switch (num) {
    default:
    case 0:  // reset
      $('#status_key').removeClass('disabled').addClass('disabled ');
      $('#status_wait').removeClass('disabled').addClass('disabled ');
      $('#status_processed').removeClass('disabled').addClass('disabled ');
      $('#status_error').removeClass('disabled').addClass('disabled ');
      break;
    case 1: // Select key
      $('#status_key').removeClass('disabled');
      $('#status_wait').removeClass('disabled').addClass('disabled ');
      $('#status_processed').removeClass('disabled').addClass('disabled ');
      $('#status_error').removeClass('disabled').addClass('disabled ');
      break;
    case 2: // Select wait
      $('#status_key').removeClass('disabled').addClass('disabled ');
      $('#status_wait').removeClass('disabled');
      $('#status_processed').removeClass('disabled').addClass('disabled ');
      $('#status_error').removeClass('disabled').addClass('disabled ');
      break;
    case 3: // Select processed
      $('#status_key').removeClass('disabled').addClass('disabled ');
      $('#status_wait').removeClass('disabled').addClass('disabled ');
      $('#status_processed').removeClass('disabled');
      $('#status_error').removeClass('disabled').addClass('disabled ');
      break;
    case 4: // Select error
      $('#status_key').removeClass('disabled').addClass('disabled ');
      $('#status_wait').removeClass('disabled').addClass('disabled ');
      $('#status_processed').removeClass('disabled').addClass('disabled ');
      $('#status_error').removeClass('disabled');
      break;
  }
}


/*
 * Transkit request
 */

function doTranskit() {
  // doReset();
  emptyResults();
  setStatus(2);
  keywords = $("#keywords").val();
  language = $("#language").val();
  host = window.location.host + "/backend";
  
  var file = document.getElementById('inputfile').files[0];
  var reader = new FileReader();

  reader.addEventListener('load', function() {
      var res = reader.result;
      const base64String = reader.result
      .replace("data:", "")
      .replace(/^.+,/, "");
      //~ console.log("base64String>>>" + base64String + "<<<");

      var send = "language=" + language + "&keywords=" + keywords + "&audio=" + base64String.replace(/\//g, "%2F").replace(/\+/g, "%2B").replace(/=/g, "%3D");
      console.log("send>>>" + send + "<<<");
      update_time();

      $.ajax({
      method: "POST",
      url: "http://"+host+"/json/",
      //url: "https://"+host+"/json/",
      data: send,
      beforeSend: function( xhr ) {
        audio.src = URL.createObjectURL(file);
        update_time();
      },
      success : function(response){
			//~ alert("SUCCESS!");
			//~ console.log(response);
			show_time();
			$("#transcription").val(response["transcription"]);
			setStatus(3);
      },
      error: function(error){
			//~ alert("ERROR!");
			console.log(error);
			throw_error();
      }
    })
});
   

  reader.readAsDataURL(file);
}

// Throw an error
function throw_error() {
  $("#status").val("Error");
  emptyResults();
  setStatus(4);
}

// Show execution time
function show_time() {
  var end = new Date().getTime();
  var time = (end - $("#start").val())/1000;
  time = Math.round(time);
  $("#op_time").text(time.toString()+"s");
}

// Update start time 
function update_time() {
  $("#start").val(new Date().getTime());
}

/*
 * Other things
 */

String.prototype.capitalize = function() {
 return this.charAt(0).toUpperCase() + this.slice(1)
}

