import Link from "next/link"
import { Button } from "@/components/ui/button"
import { FaGithub } from "react-icons/fa"

export default function Header() {
  return (
    <header className="fixed top-0 w-full bg-white/80 backdrop-blur-sm z-50 border-b border-gray-100">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <span className="text-xl font-bold bg-gradient-to-r from-red-600 to-red-500 bg-clip-text text-transparent">
            MindAI
          </span>
        </Link>
        <nav className="hidden md:flex items-center space-x-8">
          {/* GitHub Repo Button */}
          <Link href="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025" target="_blank" rel="noopener noreferrer" passHref>
            <Button
              variant="outline"
              
              className="text-gray-600 hover:text-gray-900 border border-gray-300"
            >
              <FaGithub className="mr-2" />
              GitHub Repo
            </Button>
          </Link>

          {/* Get Started Button with Beaming Border Hover Effect */}
          <Link href="https://becoming-positively-tapir.ngrok-free.app/" passHref>
            <div className="relative inline-block group">
              <Button
                variant="default"
                className="relative z-10 bg-gradient-to-r from-red-600 to-red-500 text-white"
              >
                Get Started
              </Button>
              <span
                className="pointer-events-none absolute -inset-1 rounded-lg border-2 border-red-500 
                           opacity-0 group-hover:opacity-100 transition-opacity duration-300 animate-pulse"
              ></span>
            </div>
          </Link>
        </nav>
      </div>
    </header>
  )
}

