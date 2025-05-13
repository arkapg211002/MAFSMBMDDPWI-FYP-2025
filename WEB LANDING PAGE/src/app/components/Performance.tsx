import { useState, useEffect, useRef } from "react"

export default function Performance() {
  const [isVisible, setIsVisible] = useState(false)
  const sectionRef = useRef(null)

  // Intersection Observer to trigger fade-up effect
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

  // Data for the performance cards
  const cards = [
    {
      title: "Ensemble Model Performance",
      metricLabel: "Stratified K-Fold CV",
      metricValue: "98.03%",
      width: "98.03%",
    },
    {
      title: "Hierarchical Model Performance",
      metricLabel: "Large Dataset Accuracy",
      metricValue: "96.25%",
      width: "96.25%",
    },
  ]

  return (
    <section
      id="performance"
      ref={sectionRef}
      className="py-16 md:py-24 relative bg-[#f8f8f8]"
    >
      {/* Gradient overlay at the bottom (covers 20% of section height) */}
      <div
        className="absolute bottom-0 left-0 right-0"
        style={{
          height: "20%",
          background: "linear-gradient(to top, #fff1f1, transparent)",
        }}
      />

      <div className="container mx-auto px-4 relative">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Industry-Leading Model Performance
          </h2>
          <p className="text-lg text-gray-600">
            Our ensemble model achieves exceptional accuracy through combined machine learning approaches
          </p>
        </div>
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {cards.map((card, index) => (
            <div
              key={index}
              className={`performance-card bg-white/70 backdrop-blur-sm p-6 rounded-xl border border-gray-100 transition-all ${
                isVisible ? "fadeUp" : ""
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                {card.title}
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      {card.metricLabel}
                    </span>
                    <span className="text-sm font-medium text-red-600">
                      {card.metricValue}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-red-500 to-red-600 h-2 rounded-full"
                      style={{ width: card.width }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .performance-card {
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          transition: box-shadow 0.4s ease, border-color 0.4s ease;
          opacity: 0;
          transform: translateY(50px);
        }
        .performance-card.fadeUp {
          animation: fadeUp 1s ease-out forwards;
        }
        .performance-card:hover {
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

