"use client";

import { useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { Eye, EyeOff, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";

function ResetPasswordForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  
  const [isLoading, setIsLoading] = useState(false);
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // Simple password strength calculation
  const getStrength = (pass: string) => {
    let score = 0;
    if (!pass) return { score, text: "", color: "bg-gray-200" };
    if (pass.length > 7) score += 1;
    if (/[A-Z]/.test(pass)) score += 1;
    if (/[0-9]/.test(pass)) score += 1;
    if (/[^A-Za-z0-9]/.test(pass)) score += 1;

    if (score === 0) return { score, text: "Weak", color: "bg-red-500" };
    if (score === 1) return { score, text: "Fair", color: "bg-yellow-500" };
    if (score === 2) return { score, text: "Good", color: "bg-blue-500" };
    return { score, text: "Strong", color: "bg-green-500" };
  };

  const strength = getStrength(password);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!token) {
      toast.error("Invalid or missing reset token.");
      return;
    }
    if (strength.score < 2) {
      toast.error("Please choose a stronger password");
      return;
    }
    
    setIsLoading(true);
    
    // In a real app, POST to /api/v1/auth/reset-password
    setTimeout(() => {
      setIsLoading(false);
      toast.success("Password reset successfully! Please log in.");
      router.push("/login");
    }, 1500);
  }

  if (!token) {
    return (
      <Card className="border-0 shadow-lg sm:border sm:shadow-sm text-center py-6">
        <CardHeader>
          <CardTitle className="text-xl text-red-600">Invalid Link</CardTitle>
          <CardDescription>
            This password reset link is invalid or has expired.
          </CardDescription>
          <div className="mt-4">
            <Link href="/forgot-password">
              <Button>Request a new link</Button>
            </Link>
          </div>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="border-0 shadow-lg sm:border sm:shadow-sm">
      <CardHeader className="space-y-1 text-center">
        <CardTitle className="text-2xl font-bold tracking-tight">Set new password</CardTitle>
        <CardDescription>
          Must be at least 8 characters.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="password">New Password</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                disabled={isLoading}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
                tabIndex={-1}
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            
            {password && (
              <div className="mt-1 space-y-1">
                <div className="flex h-1 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-800">
                  <motion.div
                    className={`h-full ${strength.color}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${(strength.score + 1) * 25}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-xs text-muted-foreground text-right">
                  {strength.text}
                </p>
              </div>
            )}
          </div>
          <Button type="submit" className="w-full mt-2" disabled={isLoading}>
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : null}
            Reset password
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default function ResetPasswordPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Suspense fallback={<Card className="p-8 text-center"><Loader2 className="w-6 h-6 animate-spin mx-auto" /></Card>}>
        <ResetPasswordForm />
      </Suspense>
    </motion.div>
  );
}
