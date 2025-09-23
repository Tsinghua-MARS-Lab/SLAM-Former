(function(factory) {
  //namespacing
  if(!window["Compare"]) {
    window["Compare"] = {} ;
  }
  factory(window["Compare"]) ;
})(function(Compare) { //namespace Compare

  /*** Global comparator ***/

  /* setup all elements of class compare */
  function setup_comparators() {
    var comparators = document.querySelectorAll(".compare") ;
    for(var comparator of comparators) {
      var videos = [...comparator.querySelectorAll("video")] ;
      wait_for_videos(videos, comparator, setup_comparator) ;
    }
  }

  /* wait for video metadata to be ready so that their size is known */
  function wait_for_videos(videos, comparator, action) {
    //ensure that every video is handled
    if(videos.length > 0) {
      if(videos[0].readyState < 1) {
        //the video is still not available, add hook
        videos[0].addEventListener("loadedmetadata", function(e) {
          wait_for_videos(videos.slice(1), comparator, action) ;
        }) ;
      } else {
        //the video is ready, proceed to the remaining ones
        wait_for_videos(videos.slice(1), comparator, action) ;
      }
      //manual autoplay
      if(videos[0].readyState < 4) {
        //not enough data is available, add a play trigger when available
        videos[0].addEventListener("canplay", function(e) {
          e.currentTarget.play() ;
        }) ;
      } else {
        //sufficient data is available, play
        videos[0].play() ;
      }
    } else {
      //all videos are ready, proceed to setup
      action(comparator) ;
    }
  }

  /* setup a comparator element */
  function setup_comparator(comparator) {
    //get compared element
    var compared = comparator.querySelectorAll(".compared") ;
    if(compared.length == 0) {
      console.error("comparator contains no compared element") ;
      return ;
    }

    //size
    var w = compared[0].clientWidth ;
    var h = compared[0].clientHeight ;

    //comparator style
    var sources = [] ;
    if(compared.length == 1) {
      //a single compared element, has to be vertical or horizontal
      if(compared[0].classList.contains("compared-vertical")) {
        //single image or video horizontally split
        h = h / 2 ;
        sources.push({
          elt : compared[0],
          label : compared[0].dataset.compareTop,
          x : 0,
          y : 0
        }) ;
        sources.push({
          elt : compared[0],
          label : compared[0].dataset.compareBottom,
          x : 0,
          y : h
        }) ;
      } else if(compared[0].classList.contains("compared-horizontal")) {
        //single image or video vertically split
        w = w / 2 ;
        sources.push({
          elt : compared[0],
          label : compared[0].dataset.compareLeft,
          x : 0,
          y : 0
        }) ;
        sources.push({
          elt : compared[0],
          label : compared[0].dataset.compareRight,
          x : w,
          y : 0
        }) ;
      } else {
        console.error("unable to compare a single element without compared-vertical or compared-horizontal") ;
        return ;
      }
    } else if(compared.length == 2) {
      //two compared elements, one should be left, the other one right
      var left = comparator.querySelector(".compared-left") ;
      var right = comparator.querySelector(".compared-right") ;
      if(left && right) {
        sources.push({
          elt : left,
          label : left.dataset.compare,
          x : 0,
          y : 0
        }) ;
        sources.push({
          elt : right,
          label : right.dataset.compare,
          x : 0,
          y : 0
        }) ;
      } else {
        console.error("comparator must contain a compared-left and a compared-right child") ;
        return ;
      }
    } else {
      console.error("comparison not handled for more than two elements") ;
      return ;
    }

    //Setup the div width according to the content
    comparator.style.width = w + "px" ;

    //setup visible canvas
    var canvas = document.createElement("canvas") ;
    comparator.appendChild(canvas) ;
    canvas.width = w ;
    canvas.height = h ;

    //setup context
    var context = {
      canvas : canvas,
      sources : sources,
      x : w/2, 
      y : h/2, 
      zoom : 1,
      speed : 1,
      orientation : 0,
      mag0 : create_magnifier(comparator),
      mag1 : create_magnifier(comparator)
    } ;

    //setup settings
    create_settings(comparator, context) ;

    //move split with the mouse
    comparator.addEventListener("mousemove", function(e) {
      //mouse position relative to the canvas
      var rect = context.canvas.getBoundingClientRect() ;
      var x = e.clientX - rect.x ;
      var y = e.clientY - rect.y ;

      if(x < 0 || x > context.canvas.width || y < 0 || y > context.canvas.height) {
        //the mouse is out of the canvas
        context.mag0.style.display = "none" ;
        context.mag1.style.display = "none" ;
      } else {
        // the mouse in is the canvas, update split position
        setTimeout(move_split, 0, context, x, y) ;
      }
    })

    //change split orientation
    comparator.addEventListener("click", function(e) {
      //mouse position relative to the canvas
      var rect = context.canvas.getBoundingClientRect() ;
      var x = e.clientX - rect.x ;
      var y = e.clientY - rect.y ;

      if(x >= 0 && x <= context.canvas.width && y >= 0 && y <= context.canvas.height) {
        //the mouse is in the canvas, rotate
        context.orientation = (context.orientation + 1) % 4 ;
        //schedule redraw
        setTimeout(move_split, 0, context, x, y) ;
      }
    })
    
    //hide the magnifiers on mouse leaving the canvas
    comparator.addEventListener("mouseleave", function(e) {
      context.mag0.style.display = "none" ;
      context.mag1.style.display = "none" ;
    })

    //splatting
    if(sources[0].elt.nodeName == "VIDEO" || sources[1].elt.nodeName == "VIDEO") {
      //for videos schedule regular splatting
      setInterval(transfer_compare, 40, context) ;
      setInterval(transfer_magnifiers, 40, context) ;
    } else {
      //for images splat once, update handled by mouseover
      setTimeout(transfer_compare, 0, context) ;
      setTimeout(transfer_magnifiers, 0, context) ;
    }
  }

  /* move the split to a given relative position */
  function move_split(context, x, y) {
    //position relative to the canvas
    context.x = x ;
    context.y = y ;

    //update magnifier position
    move_magnifiers(context, x, y) ;

    //request redraw
    setTimeout(transfer_compare, 0, context) ;
    setTimeout(transfer_magnifiers, 0, context) ;
  }

  /* splatting on the canvas */
  function transfer_compare(context) {
    //context information
    var ctx = context.canvas.getContext("2d") ;
    var w = context.canvas.width ;
    var h = context.canvas.height ;
    var s = context.sources ;

    //splat first image on the whole canvas
    ctx.drawImage(s[0].elt, s[0].x, s[0].y, w, h, 0, 0, w, h) ;

    //separator color
    ctx.strokeStyle = "#FF0000" ;

    //splat the second image
    if(context.orientation % 2 == 0) {
      //vertical split position
      var x = Math.min(Math.max(1, context.x), w-1) ;

      //split sides
      if(context.orientation == 0) {
        //other image on the right
        ctx.drawImage(s[1].elt, s[1].x + x, s[1].y, w - x, h, x, 0, w - x, h) ;
      } else {
        //other image on the left
        ctx.drawImage(s[1].elt, s[1].x, s[1].y, x, h, 0, 0, x, h) ;
      }

      //separator
      ctx.beginPath() ;
      ctx.moveTo(x, 0) ;
      ctx.lineTo(x, h) ;
      ctx.stroke() ;
    } else {
      //horizontal split position
      var y = Math.min(Math.max(1, context.y), h-1) ;

      //split sides
      if(context.orientation == 1) {
        //other image on top
        ctx.drawImage(s[1].elt, s[1].x, s[1].y, w, y, 0, 0, w, y) ;
      } else {
        //other image below
        ctx.drawImage(s[1].elt, s[1].x , s[1].y + y, w, h - y, 0, y, w, h - y) ;
      }

      //separator
      ctx.beginPath() ;
      ctx.moveTo(0, y) ;
      ctx.lineTo(w, y) ;
      ctx.stroke() ;
    }
  }

  /*** Magnifiers ***/

  /* add magnifiers to the DOM */
  function create_magnifier(elt) {
    //div container for cropping and margins
    var mag = document.createElement("div") ;
    mag.className = "magnifier" ;
    mag.style.display = "none" ;
    elt.appendChild(mag) ;

    //canvas to splat the images
    var canvas = document.createElement("canvas") ;
    mag.appendChild(canvas) ;

    return mag ;
  }

  /* place the magnifiers around the cursor */
  function move_magnifiers(context, x, y) {
    //magnifier canvases
    var m0 = context.mag0 ;
    var m1 = context.mag1 ;

    //ensure magnifiers are shown
    m0.style.display = "block" ;
    m1.style.display = "block" ;

    //change x and y to the comparator frame
    var canvas_rect = context.canvas.getBoundingClientRect() ;
    var comparator_rect = context.canvas.closest(".compare").getBoundingClientRect() ;
    x = x + canvas_rect.left - comparator_rect.left ;
    y = y + canvas_rect.top - comparator_rect.top ;

    //positionning around cursor
    if(context.orientation % 2 == 0) {
      //vertical split
      var w = comparator_rect.width ;
      if(context.orientation == 0) {
        //m0 left, m1 right
        m0.style.left = "" ;
        m0.style.right = w - x + "px" ;
        m1.style.right = "" ;
        m1.style.left = x + "px" ;
      } else {
        //m0 right, m1 left
        m0.style.right = "" ;
        m0.style.left = x + "px" ;
        m1.style.left = "" ;
        m1.style.right = w - x + "px" ;
      }
      //align vertically with the cursor
      m0.style.bottom = "" ;
      m1.style.bottom = "" ;
      m0.style.top = y - m0.offsetHeight / 2 + "px" ;
      m1.style.top = y - m1.offsetHeight / 2 + "px" ;
    } else {
      //horizontal split
      var h = comparator_rect.height ;
      if(context.orientation == 1) {
        //m0 below, m1 above
        m0.style.bottom = "" ;
        m0.style.top = y + "px" ;
        m1.style.top = "" ;
        m1.style.bottom = h - y + "px" ;
      } else {
        //m0 above, m1 below
        m0.style.top = "" ;
        m0.style.bottom = h - y + "px" ;
        m1.style.bottom = "" ;
        m1.style.top = y + "px" ;
      }
      //align horizontally with cursor
      m0.style.right = "" ;
      m1.style.right = "" ;
      m0.style.left = x - m0.offsetWidth / 2 + "px" ;
      m1.style.left = x - m1.offsetWidth / 2 + "px" ;
    }
  }

  /* splatting on the magnifiers */
  function transfer_magnifiers(context) {
    //ensure that the magnifiers are visible
    if(context.mag0.style.display != "none") {
      //magnifier contexts
      var ctx0 = context.mag0.querySelector("canvas").getContext("2d") ;
      var ctx1 = context.mag1.querySelector("canvas").getContext("2d") ;

      //cursor position
      var x = context.x ;
      var y = context.y ;

      //splatting area
      var ctx_x = 0 ;
      var ctx_y = 0 ;
      var ctx_w = context.mag0.clientWidth ;
      var ctx_h = context.mag0.clientHeight ;

      //clear and disable pixel smoothing
      for(var ctx of [ctx0, ctx1]) {
        ctx.canvas.width = ctx_w ;
        ctx.canvas.height = ctx_h ;
        ctx.fillStyle = "#494949" ;
        ctx.beginPath() ;
        ctx.fillRect(0, 0, ctx_w, ctx_h) ;
        ctx.stroke() ;
        ctx.imageSmoothingEnabled = false ;
      }

      //source boundaries
      var max_x = context.canvas.width ;
      var max_y = context.canvas.height;

      //source window
      var scale = context.zoom ;
      var im_x = x - ctx_w / (2*scale) ;
      var im_w = ctx_w / scale ;
      var im_y = y - ctx_h / (2*scale) ;
      var im_h = ctx_h / scale ;

      //boundary handling
      if(im_x < 0) {
        //clamp to the left
        ctx_x = - scale * im_x ;
        im_w = im_w + im_x ;
        im_x = 0 ;
      }
      if(im_x + im_w > max_x) {
        //clamp to the right
        ctx_w = scale * (max_x - im_x) ;
        im_w = max_x - im_x ;
      }

      if(im_y < 0) {
        //clamp to the top
        ctx_y = - scale * im_y ;
        im_h = im_h + im_y ;
        im_y = 0 ;
      }
      if(im_y + im_h > max_y) {
        //clamp to the bottom
        ctx_h = scale * (max_y - im_y) ;
        im_h = max_y - im_y ;
      }

      //splat
      var s = context.sources ;
      if(im_w > 0 && im_h > 0) {
        ctx0.drawImage(
          s[0].elt, 
          s[0].x + im_x, s[0].y + im_y, im_w, 
          im_h, ctx_x, ctx_y, ctx_w, ctx_h
        ) ;
        ctx1.drawImage(
          s[1].elt, 
          s[1].x + im_x, s[1].y + im_y, im_w, im_h, 
          ctx_x, ctx_y, ctx_w, ctx_h
        ) ;
      }

      //labels
      if(s[0].label) {
        ctx0.font = "20px Arial" ;
        ctx0.strokeStyle = "#000" ;
        ctx0.strokeText(s[0].label, 10, 20) ;
        ctx0.fillStyle = "#fff" ;
        ctx0.fillText(s[0].label, 10, 20) ;
      }
      if(s[1].label) {
        ctx1.font = "20px Arial" ;
        ctx1.strokeStyle = "#000" ;
        ctx1.strokeText(s[1].label, 10, 20) ;
        ctx1.fillStyle = "#fff" ;
        ctx1.fillText(s[1].label, 10, 20) ;
      }
    }
  }

  /*** Settings ***/

  /* slider creation helper */
  function create_slider(name, context, context_target, min, max, step) {
    //container for everything
    var container = document.createElement("div") ;
    container.className = "compare-setting" ;

    //slider
    var input = document.createElement("input") ;
    input.type = "range" ;
    input.min = min ;
    input.max = max ;
    if(step) {
      input.step = step ;
    }
    input.value = context[context_target] ;

    //frame around the slider
    var frame = document.createElement("div") ;
    frame.className = "range-container" ;
    
    //slider name
    var label = document.createElement("label") ;
    label.innerHTML = name ;
    
    //slider value
    var value = document.createElement("label") ;
    value.innerHTML = input.value ;

    //put everything together
    container.appendChild(label) ;
    frame.appendChild(input) ;
    container.appendChild(frame) ;
    container.appendChild(value) ;

    //update the value label and the context variable
    input.addEventListener("change", function(e) {
      var v = e.currentTarget.value ;
      value.innerHTML = v ;
      context[context_target] = v ;
    }) ;

    return container ;
  }

  function create_play_toggle(context) {
    //container
    var container = document.createElement("div") ;
    container.className = "compare-setting" ;

    //play / pause
    var icon = document.createElement("div") ;
    icon.className = "pause" ;
    var videos = context.sources.map(c => c.elt).filter(e => e.nodeName == "VIDEO") ;
    icon.addEventListener("click", function() {
      if(icon.classList.contains("play")) {
        icon.classList.replace("play", "pause") ;
        videos.forEach(v => v.play()) ;
      } else {
        icon.classList.replace("pause", "play") ;
        videos.forEach(v => v.pause()) ;
      }
    }) ;

    container.appendChild(icon) ;
    return container ;
  }

  /* add settings bar to the DOM */
  function create_settings(elt, context) {
    //settings bar
    var settings = document.createElement("div") ;
    settings.className = "compare-settings" ;
    elt.appendChild(settings) ;

    //zoom slider
    var zoom = create_slider("Zoom", context, "zoom", 1, 10) ;
    settings.appendChild(zoom) ;

    //video specific settings
    var c0 = context.sources[0].elt ;
    var c1 = context.sources[1].elt ;
    if(c0.nodeName == "VIDEO" || c1.nodeName == "VIDEO") {
      //play pause
      var playpause = create_play_toggle(context) ;
      settings.appendChild(playpause) ;

      //video speed slider
      var speed = create_slider("Speed", context, "speed", 0, 1, 0.1)
      var slider = speed.querySelector("input")
      //update video playback speed on slider change
      slider.addEventListener("change", function(e) {
        if(c0.nodeName == "VIDEO") {
          c0.playbackRate = e.currentTarget.value ;
        }
        if(c1.nodeName == "VIDEO") {
          c1.playbackRate = e.currentTarget.value ;
        }
      }) ;
      settings.appendChild(speed) ;
    }

    return zoom ;
  }

  /*** trigger ***/

  window.addEventListener("load", setup_comparators) ;
})
