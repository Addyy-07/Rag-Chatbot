"use client";

import { useState } from "react";
import Script from "next/script";
import { motion } from "framer-motion";
import { Loader2, CreditCard, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

export default function CheckoutPage() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handlePayment = async () => {
    setIsLoading(true);
    
    // Step 1: Call backend to create order
    try {
      // Assuming user is logged in and token is in localStorage for this example, 
      // or we'd fetch it from context/cookies.
      const token = localStorage.getItem("token") || ""; // Adjust auth token retrieval as needed

      const orderRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/billing/create-order`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          amount: 90000, // $9.00 or 900.00 INR in paise
          currency: "INR"
        })
      });

      if (!orderRes.ok) {
        throw new Error("Failed to create order");
      }

      const orderData = await orderRes.json();

      // Step 2: Open Razorpay Checkout Modal
      const options = {
        key: process.env.NEXT_PUBLIC_RAZORPAY_KEY_ID, // Enter the Key ID generated from the Dashboard
        amount: orderData.amount.toString(),
        currency: orderData.currency,
        name: "DocChat AI",
        description: "Pro Tier Subscription",
        image: "https://your-logo-url.com/logo.png",
        order_id: orderData.order_id,
        handler: async function (response: any) {
          // Step 3: Verify Payment Signature on Backend
          try {
            const verifyRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/billing/verify-payment`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
              },
              body: JSON.stringify({
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature
              })
            });

            if (!verifyRes.ok) {
              throw new Error("Payment verification failed");
            }

            toast.success("Payment successful! You are now on the Pro tier.");
            router.push("/dashboard"); // Or wherever appropriate
          } catch (error) {
            console.error("Verification error:", error);
            toast.error("Payment verification failed. Please contact support.");
          }
        },
        prefill: {
          name: "User",
          email: "user@example.com",
          contact: "9999999999"
        },
        notes: {
          address: "DocChat AI Corporate Office"
        },
        theme: {
          color: "#0f172a" // slate-900 (primary color)
        }
      };

      // @ts-ignore - Razorpay is loaded via script
      const rzp1 = new window.Razorpay(options);
      
      rzp1.on('payment.failed', function (response: any) {
        console.error(response.error);
        toast.error(`Payment failed: ${response.error.description}`);
      });

      rzp1.open();
    } catch (error) {
      console.error(error);
      toast.error("An error occurred during checkout setup.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Script src="https://checkout.razorpay.com/v1/checkout.js" strategy="lazyOnload" />
      
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-zinc-950 p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-md"
        >
          <Card className="border-0 shadow-lg sm:border sm:shadow-sm">
            <CardHeader className="space-y-1 text-center bg-primary/5 rounded-t-xl pb-8 pt-6 border-b">
              <div className="mx-auto bg-primary/10 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <ShieldCheck className="w-6 h-6 text-primary" />
              </div>
              <CardTitle className="text-2xl font-bold tracking-tight">Upgrade to Pro</CardTitle>
              <CardDescription className="text-base mt-2">
                Secure checkout powered by Razorpay
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-muted-foreground">Pro Tier Subscription (1 Month)</span>
                  <span className="font-semibold">₹900.00</span>
                </div>
                <div className="flex justify-between items-center py-2 text-lg font-bold">
                  <span>Total</span>
                  <span>₹900.00</span>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex flex-col gap-4">
              <Button 
                onClick={handlePayment} 
                className="w-full h-12 text-lg font-semibold" 
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                ) : (
                  <CreditCard className="mr-2 h-5 w-5" />
                )}
                Pay Securely
              </Button>
              <p className="text-xs text-center text-muted-foreground">
                By clicking pay, you agree to our Terms of Service.
              </p>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </>
  );
}
