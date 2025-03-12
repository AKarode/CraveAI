import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <>
      <div className="container py-12 mx-auto">
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-2">
          <div className="flex flex-col justify-center space-y-6">
            <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
              Get personalized menu recommendations
            </h1>
            <p className="max-w-[600px] text-muted-foreground md:text-xl">
              Scan any menu, tell us your dietary preferences, and get personalized recommendations tailored just for you.
            </p>
            <div className="flex flex-col gap-3 min-[400px]:flex-row">
              <Link href="/scan">
                <Button size="lg" className="w-full">Scan a Menu</Button>
              </Link>
              <Link href="/chat">
                <Button size="lg" variant="outline" className="w-full">Chat with AI</Button>
              </Link>
            </div>
          </div>
          <div className="flex items-center justify-center">
            <div className="relative h-[350px] w-[350px] sm:h-[400px] sm:w-[400px] md:h-[450px] md:w-[450px] lg:h-[500px] lg:w-[500px]">
              <div className="absolute inset-0 bg-gradient-to-r from-primary to-primary/60 rounded-full opacity-70 blur-3xl" />
              <div className="relative h-full w-full bg-muted rounded-lg border shadow-lg overflow-hidden flex items-center justify-center">
                <p className="text-xl font-medium text-center p-4">Menu scanning and AI recommendations</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-20 grid gap-8 md:grid-cols-3">
          <div className="rounded-lg border bg-card p-8 shadow-sm">
            <h3 className="text-xl font-bold">Scan Menus</h3>
            <p className="mt-2 text-muted-foreground">
              Use your phone camera to scan restaurant menus or upload PDF menus.
            </p>
          </div>
          <div className="rounded-lg border bg-card p-8 shadow-sm">
            <h3 className="text-xl font-bold">Personalized Recommendations</h3>
            <p className="mt-2 text-muted-foreground">
              Get dish recommendations based on your dietary preferences and restrictions.
            </p>
          </div>
          <div className="rounded-lg border bg-card p-8 shadow-sm">
            <h3 className="text-xl font-bold">Chat with AI</h3>
            <p className="mt-2 text-muted-foreground">
              Ask questions about menu items, ingredients, and get detailed information.
            </p>
          </div>
        </div>
      </div>

      <footer className="border-t py-6 md:py-0 mt-16">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-16 md:flex-row">
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Crave AI. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <Link href="/terms" className="text-sm text-muted-foreground underline-offset-4 hover:underline">
              Terms
            </Link>
            <Link href="/privacy" className="text-sm text-muted-foreground underline-offset-4 hover:underline">
              Privacy
            </Link>
          </div>
        </div>
      </footer>
    </>
  );
}
