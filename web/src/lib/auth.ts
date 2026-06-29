import { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      if (account?.provider === "google") {
        try {
          // Send the Google ID token to our FastAPI backend
          const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/google`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              id_token: account.id_token,
            }),
          });

          if (!res.ok) {
            console.error("Backend OAuth verification failed");
            return false;
          }

          const data = await res.json();
          // Store the backend JWT and user data in the session
          // We attach it to the user object temporarily so jwt callback can access it
          (user as any).backendToken = data.access_token;
          (user as any).backendUser = data.user;
          
          return true;
        } catch (error) {
          console.error("Error exchanging token with backend:", error);
          return false;
        }
      }
      return true; // Other providers (if any) or credentials
    },
    async jwt({ token, user }) {
      // If user object exists (only on initial sign in), attach our backend data to the token
      if (user) {
        token.accessToken = (user as any).backendToken;
        token.user = (user as any).backendUser;
      }
      return token;
    },
    async session({ session, token }) {
      // Pass the backend token and user data to the client session
      (session as any).accessToken = token.accessToken;
      (session as any).user = token.user;
      return session;
    },
  },
  pages: {
    signIn: "/login",
  },
  session: {
    strategy: "jwt",
  },
};
