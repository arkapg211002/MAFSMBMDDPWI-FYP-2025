import { useState, useEffect, useRef } from "react"
import {
  MessageSquare,
  ImageIcon,
  FileText,
  Globe,
  Brain,
  Camera,
  Languages,
  BarChart3,
  RefreshCw,
  ClipboardCheck,
} from "lucide-react"

export default function Features() {
  const features = [
    {
      icon: <MessageSquare className="h-6 w-6" />,
      title: "Text Analysis",
      description:
        "Detect mental health issues from textual inputs with advanced NLP",
    },
    {
      icon: <ImageIcon className="h-6 w-6" />,
      title: "Image & Video Analysis",
      description:
        "Extract insights from visual content using computer vision",
    },
    {
      icon: <FileText className="h-6 w-6" />,
      title: "PDF Processing",
      description: "Convert and analyze typed or handwritten texts",
    },
    {
      icon: <Globe className="h-6 w-6" />,
      title: "Social Media Analysis",
      description:
        "Analyze posts from Reddit and Twitter for mental health indicators",
    },
    {
      icon: <Brain className="h-6 w-6" />,
      title: "User response to Image",
      description:
        "Digital implementation of psychological image interpretation",
    },
    {
      icon: <Languages className="h-6 w-6" />,
      title: "Multilingual Support",
      description: "Analyze content across multiple languages",
    },
    {
      icon: <BarChart3 className="h-6 w-6" />,
      title: "Wellbeing Mapping",
      description: "Use Ryff's Scale to assess psychological wellbeing",
    },
    {
      icon: <RefreshCw className="h-6 w-6" />,
      title: "Model Retraining",
      description: "Continuously improve with new data",
    },
    {
      icon: <ClipboardCheck className="h-6 w-6" />,
      title: "Wellbeing Survey",
      description: "Complete assessment based on Ryff's Scale",
    },
  ]

  // Intersection Observer state for triggering the fade-up effect
  const [isVisible, setIsVisible] = useState(false)
  const sectionRef = useRef(null)

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
    if (sectionRef.current) {
      observer.observe(sectionRef.current)
    }
    return () => {
      if (sectionRef.current) observer.unobserve(sectionRef.current)
    }
  }, [])

  return (
    <section
      id="features"
      ref={sectionRef}
      className="py-16 md:py-24 relative overflow-hidden"
    >
      {/* Linear background gradient: reddish only on the far right (15%) then light grey */}
      <div
        className="absolute inset-0"
        style={{
          background: "linear-gradient(to left, #fff1f1 15%, #f8f8f8 100%)",
        }}
      />

      <div className="container mx-auto px-4 relative">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Comprehensive Mental Health Analysis
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our AI framework combines multiple analysis methods to provide accurate
            mental health insights
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`feature-card bg-white/70 backdrop-blur-sm p-6 rounded-2xl border border-gray-100 transition-all ${
                isVisible ? "fadeUp" : ""
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="w-12 h-12 bg-gradient-to-br from-rose-50 to-rose-100 rounded-xl flex items-center justify-center mb-4">
                <div className="text-rose-600">{feature.icon}</div>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .feature-card {
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          /* Start hidden for animation */
          opacity: 0;
          transform: translateY(50px);
        }
        .feature-card.fadeUp {
          animation: fadeUp 1s ease-out forwards;
        }
        .feature-card:hover {
          border-color: rgba(253, 232, 232, 0.6);
          box-shadow: 0 6px 12px rgba(253, 232, 232, 0.8);
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
      `}</style>
    </section>
  )
}

