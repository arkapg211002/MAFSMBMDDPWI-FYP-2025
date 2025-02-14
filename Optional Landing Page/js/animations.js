// Brain animation SVG manipulation
document.addEventListener('DOMContentLoaded', () => {
    const brainAnimation = document.querySelector('.brain-animation');

    // Create SVG neural network visualization
    const createNeuralNetwork = () => {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 200 200");
        svg.style.width = "300px";
        svg.style.height = "300px";

        // Add nodes
        for (let i = 0; i < 20; i++) {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", Math.random() * 200);
            circle.setAttribute("cy", Math.random() * 200);
            circle.setAttribute("r", "3");
            circle.setAttribute("fill", "#ff8080"); // Light red color
            svg.appendChild(circle);
        }

        // Add connections
        for (let i = 0; i < 30; i++) {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", Math.random() * 200);
            line.setAttribute("y1", Math.random() * 200);
            line.setAttribute("x2", Math.random() * 200);
            line.setAttribute("y2", Math.random() * 200);
            line.setAttribute("stroke", "#ff8080"); // Light red color
            line.setAttribute("stroke-width", "0.5");
            line.setAttribute("opacity", "0.5");
            svg.appendChild(line);
        }

        brainAnimation.appendChild(svg);
    };

    createNeuralNetwork();

    // Animate nodes
    const animateNodes = () => {
        const nodes = document.querySelectorAll('circle');
        nodes.forEach(node => {
            const currentCx = parseFloat(node.getAttribute('cx'));
            const currentCy = parseFloat(node.getAttribute('cy'));

            node.setAttribute('cx', currentCx + (Math.random() - 0.5) * 2);
            node.setAttribute('cy', currentCy + (Math.random() - 0.5) * 2);
        });

        requestAnimationFrame(animateNodes);
    };

    animateNodes();

    // Carousel functionality
    const carousel = document.querySelector('.carousel');
    const slides = document.querySelectorAll('.carousel-slide');
    let currentSlide = 0;

    function moveToSlide(index) {
        if (carousel) {
            carousel.style.transform = `translateX(-${index * 100}%)`;
        }
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % slides.length;
        moveToSlide(currentSlide);
    }

    // Auto-advance carousel every 5 seconds
    setInterval(nextSlide, 5000);
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scroll = window.pageYOffset;
    const hero = document.querySelector('.hero-content');
    hero.style.transform = `translateY(${scroll * 0.5}px)`;
});

// Floating animation for feature cards on hover
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.animation = 'float 3s ease-in-out infinite';
    });

    card.addEventListener('mouseleave', () => {
        card.style.animation = 'none';
    });
});