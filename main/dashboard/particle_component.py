import streamlit.components.v1 as components

def render_particles():
    """
    Automatically renders particles that adjust based on the app's light/dark theme.
    """
    particles_js_code = """
    <div id="particles-js"></div>
    <style>
      #particles-js {
        position: absolute;
        width: 100%;
        height: 100vh;
        background-color: transparent; /* Transparent background */
        z-index: -1;
      }
    </style>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
      // Function to apply particles dynamically based on the theme
      function applyParticles() {
        const isDarkMode = getComputedStyle(document.body).backgroundColor === 'rgb(18, 18, 18)'; // Detect dark mode
        const particleColor = isDarkMode ? '#87ffff' : '#1e90ff'; // Aqua blue for dark, Dodger blue for light
        const lineColor = isDarkMode ? '#00008b' : '#87CEFA'; // Dark blue for dark, Light blue for light

        particlesJS("particles-js", {
          "particles": {
            "number": {
              "value": 80,
              "density": {
                "enable": true,
                "value_area": 800
              }
            },
            "color": {
              "value": particleColor /* Particle color based on mode */
            },
            "shape": {
              "type": "circle",
              "stroke": {
                "width": 0,
                "color": "#000000"
              },
              "polygon": {
                "nb_sides": 4
              }
            },
            "opacity": {
              "value": 1.52,
              "random": false,
              "anim": {
                "enable": false,
                "speed": 1,
                "opacity_min": 0.1,
                "sync": false
              }
            },
            "size": {
              "value": 2.7,
              "random": true,
              "anim": {
                "enable": false,
                "speed": 40,
                "size_min": 0.1,
                "sync": false
              }
            },
            "line_linked": {
              "enable": true,
              "distance": 150,
              "color": lineColor, /* Line color based on mode */
              "opacity": 0.4,
              "width": 1
            },
            "move": {
              "enable": true,
              "speed": 1.5,
              "direction": "none",
              "random": false,
              "straight": false,
              "out_mode": "out",
              "bounce": false,
              "attract": {
                "enable": false,
                "rotateX": 600,
                "rotateY": 1200
              }
            }
          },
          "interactivity": {
            "detect_on": "canvas",
            "events": {
              "onhover": {
                "enable": true,
                "mode": "grab"
              },
              "onclick": {
                "enable": true,
                "mode": "push"
              },
              "resize": true
            },
            "modes": {
              "grab": {
                "distance": 140,
                "line_linked": {
                  "opacity": 1
                }
              },
              "bubble": {
                "distance": 400,
                "size": 40,
                "duration": 2,
                "opacity": 8,
                "speed": 3
              },
              "repulse": {
                "distance": 200,
                "duration": 0.4
              },
              "push": {
                "particles_nb": 4
              },
              "remove": {
                "particles_nb": 2
              }
            }
          },
          "retina_detect": true
        });
      }

      // Initialize particles when the page loads
      document.addEventListener('DOMContentLoaded', applyParticles);
    </script>
    """

    # Display the HTML
    components.html(particles_js_code, height=750)