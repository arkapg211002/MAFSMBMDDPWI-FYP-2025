import Header from "./components/Header"
import Hero from "./components/Hero"
import Features from "./components/Features"
import Performance from "./components/Performance"
import Footer from "./components/Footer"

export default function Home() {
  return (
    <div className="min-h-screen relative">
      {/* Background gradients */}
      <div className="absolute inset-0 bg-gradient-to-b from-gray-50 to-white -z-10" />
      <div className="absolute top-0 left-0 w-full h-[800px] bg-gradient-to-br from-red-50/50 via-gray-50/50 to-transparent -z-10" />
      <div className="absolute top-[20%] right-0 w-[500px] h-[500px] bg-gradient-to-bl from-red-100/20 via-gray-100/20 to-transparent rounded-full blur-3xl -z-10" />
      <div className="absolute top-[60%] left-0 w-[500px] h-[500px] bg-gradient-to-tr from-red-50/30 via-gray-100/30 to-transparent rounded-full blur-3xl -z-10" />

      <Header />
      <main>
        <Hero />
        <Features />
        <Performance />
      </main>
      <Footer />
    </div>
  )
}

