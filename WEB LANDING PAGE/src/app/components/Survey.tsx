import { Button } from "@/components/ui/button"

export default function Survey() {
  return (
    <section id="survey" className="py-16 md:py-24 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Take the Wellbeing Assessment</h2>
          <p className="text-lg text-gray-600 mb-8">
            Complete our comprehensive survey based on Ryff's Psychological Wellbeing Scale to receive personalized
            insights
          </p>
          <div className="bg-white p-8 rounded-xl shadow-sm">
            <div className="max-w-md mx-auto">
              <p className="text-gray-600 mb-6">
                The assessment takes approximately 15 minutes to complete and provides detailed insights into your
                psychological wellbeing across multiple dimensions.
              </p>
              <Button size="lg" className="w-full bg-gradient-to-r from-red-600 to-red-500">
                Start Assessment
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

