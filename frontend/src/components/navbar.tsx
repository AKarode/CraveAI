import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/mode-toggle";

export function Navbar() {
  return (
    <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between mx-auto">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl">
          <span className="text-primary">Crave</span>AI
        </Link>
        <nav className="flex items-center gap-4">
          <Link href="/profile">
            <Button variant="ghost">Profile</Button>
          </Link>
          <Link href="/scan">
            <Button variant="ghost">Scan Menu</Button>
          </Link>
          <Link href="/chat">
            <Button variant="default">Chat</Button>
          </Link>
          <ModeToggle />
        </nav>
      </div>
    </header>
  );
} 