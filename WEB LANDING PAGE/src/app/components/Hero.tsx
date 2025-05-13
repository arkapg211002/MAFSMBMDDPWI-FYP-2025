import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { FaPlay } from "react-icons/fa"

export default function Hero() {
  // Array of 9 image URLs (replace these with your actual image URLs)
  const images = [
    "/assets/images/image1.png",
    "/assets/images/image2.png",
    "/assets/images/image3.png",
    "/assets/images/image4.png",
    "/assets/images/image5.png",
    "/assets/images/image6.png",
    "/assets/images/image7.png",
    "/assets/images/image8.png",
    "/assets/images/image9.png",
  ]

  const [currentIndex, setCurrentIndex] = useState(0)
  const [prevIndex, setPrevIndex] = useState(null)
  const [tiltY, setTiltY] = useState(0)
  const [isVisible, setIsVisible] = useState(false)
  const carouselRef = useRef(null)

  // Update current image every 3 seconds.
  // When updating, store the current index as the previous index.
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentIndex((prev) => {
        setPrevIndex(prev)
        return (prev + 1) % images.length
      })
    }, 3000)
    return () => clearInterval(intervalId)
  }, [images.length])

  // Clear the previous image after the fade-out animation (1s)
  useEffect(() => {
    if (prevIndex !== null) {
      const timer = setTimeout(() => {
        setPrevIndex(null)
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [prevIndex])

  // Handle dynamic tilting based on mouse position
  const handleMouseMove = (e) => {
    if (carouselRef.current) {
      const rect = carouselRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const deltaX = e.clientX - centerX
      const maxTilt = 5 // maximum tilt in degrees
      const r = deltaX / (rect.width / 2)
      setTiltY(r * maxTilt)
    }
  }

  const handleMouseLeave = () => {
    setTiltY(0)
  }

  // Intersection Observer to trigger the fade-up animation when carousel is in view
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true)
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1 }
    )
    if (carouselRef.current) {
      observer.observe(carouselRef.current)
    }
    return () => {
      if (carouselRef.current) observer.unobserve(carouselRef.current)
    }
  }, [])

  return (
    <section className="pt-32 pb-16 md:pt-40 md:pb-24 relative overflow-hidden">
      {/* Decorative gradient circles */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-br from-red-100/20 via-gray-100/20 to-transparent rounded-full blur-3xl -z-10" />

      <div className="container mx-auto px-4 relative">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-block rounded-full bg-gradient-to-r from-red-50 to-red-100 px-4 py-1.5 mb-6">
            <span className="text-sm font-medium bg-gradient-to-r from-red-600 to-red-500 bg-clip-text text-transparent">
              AI-Powered Mental Health Analysis
            </span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-8 bg-gradient-to-r from-gray-900 to-gray-400 bg-clip-text text-transparent">
            Multimodal AI Framework for Mental Health Detection
          </h1>
          <p className="text-lg md:text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
            Leveraging social media data and advanced AI to detect mental health disorders early, enabling timely
            intervention and personalized wellbeing insights.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            {/* Try Now Button with Icon and External Link */}
            <a href="https://becoming-positively-tapir.ngrok-free.app/">
              <Button
                size="lg"
                className="bg-gradient-to-r from-red-600 to-red-500 text-white hover:from-red-500 hover:to-red-600 transition-all duration-200 shadow-lg"
              >
                <FaPlay className="mr-2" />
                Try Now
              </Button>
            </a>

            {/* Download Sample Input Button with Download Attribute */}
            <a
              href="https://drive.google.com/uc?export=download&id=1VH3QKcllymxj3CSeRkWBe9XbN0PTrvs8"
              download
            >
              <Button
                size="lg"
                variant="outline"
                className="border-red-200 hover:bg-red-50 transition-all duration-200 shadow-lg"
              >
                Download Sample Input
              </Button>
            </a>
          </div>

          {/* Carousel Section */}
          <div
            ref={carouselRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className={`group mt-12 relative mx-auto w-full max-w-4xl aspect-[2/1] ${isVisible ? "fadeUp" : ""}`}
            style={{ perspective: "1000px" }}
          >
            {/* Previous Image: fades out */}
            {prevIndex !== null && (
              <img
                src={images[prevIndex]}
                alt={`Slide ${prevIndex + 1}`}
                className="absolute inset-0 w-full h-full object-contain rounded-lg shadow-2xl fadeOut"
                style={{
                  transform: `rotateX(10deg) rotateY(${tiltY}deg)`,
                }}
              />
            )}
            {/* Current Image: fades in */}
            <img
              key={currentIndex}
              src={images[currentIndex]}
              alt={`Slide ${currentIndex + 1}`}
              className="relative w-full h-full object-contain rounded-lg shadow-2xl fadeIn"
              style={{
                transform: `rotateX(10deg) rotateY(${tiltY}deg)`,
              }}
            />
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        .fadeIn {
          animation: fadeIn 1s ease-in-out forwards;
        }
        @keyframes fadeOut {
          from {
            opacity: 1;
          }
          to {
            opacity: 0;
          }
        }
        .fadeOut {
          animation: fadeOut 1s ease-in-out forwards;
        }
        @keyframes fadeUp {
          from {
            opacity: 0;
            transform: translateY(50px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .fadeUp {
          animation: fadeUp 1s ease-out forwards;
        }
      `}</style>
    </section>
  )
}

