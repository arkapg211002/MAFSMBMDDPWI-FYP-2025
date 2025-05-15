import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'v13',
  description: 'AI powered mental health disorder detection',
  icons: {
    icon: './favicon.ico', // Path to your favicon file in the public folder
  },

}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>{children}</body>
    </html>
  )
}
