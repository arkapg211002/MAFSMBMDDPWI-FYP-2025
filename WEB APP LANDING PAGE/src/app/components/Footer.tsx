import Link from "next/link"

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-300 py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-white font-bold mb-4">MindAI</h3>
            <p className="text-sm">AI-powered mental health analysis and wellbeing insights platform</p>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Features</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="#" className="hover:text-white">
                  Text Analysis
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Image Analysis
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Social Media Analysis
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Wellbeing Survey
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="#" className="hover:text-white">
                  Documentation
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  API Reference
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Research Papers
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Case Studies
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Legal</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="#" className="hover:text-white">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Data Protection
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-white">
                  Cookie Policy
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-sm text-center">
          <p>&copy; {new Date().getFullYear()} MindAI. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

