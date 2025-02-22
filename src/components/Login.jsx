import React, { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGoogle, faFacebookF } from "@fortawesome/free-brands-svg-icons";
import { useNavigate } from "react-router-dom";  // Import useNavigate

const Login = () => {
  const [isActive, setIsActive] = useState(false);
  const navigate = useNavigate();  // Initialize navigate function

  const handleRegisterClick = () => {
    setIsActive(true);
  };

  const handleLoginClick = () => {
    setIsActive(false);
  };

  const handleSignUp = () => {
    // You can include any sign-up logic here before navigating
    navigate("/calendar");  // Redirect to /calendar route
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F7F6F2]">
      <div className="relative bg-white rounded-3xl shadow-lg overflow-hidden w-[768px] max-w-full min-h-[480px] flex">
        
        {/* Sign In Form */}
        <div
          className={`form-container w-1/2 flex flex-col justify-center items-center px-10 bg-white transition-transform duration-700 ${
            isActive ? "-translate-x-full" : "translate-x-0"
          }`}
        >
          <form className="flex flex-col justify-center items-center w-full">
            <h1 className="text-3xl font-bold mb-4">Sign In</h1>
            <div className="flex justify-center mb-4">
              <a
                href="#"
                className="flex items-center justify-center w-10 h-10 border border-gray-400 rounded-full mx-2 hover:bg-[#006953] hover:text-white transition-all"
              >
                <FontAwesomeIcon icon={faGoogle} />
              </a>
              <a
                href="#"
                className="flex items-center justify-center w-10 h-10 border border-gray-400 rounded-full mx-2 hover:bg-[#006953] hover:text-white transition-all"
              >
                <FontAwesomeIcon icon={faFacebookF} />
              </a>
            </div>
            <span className="text-sm text-gray-600 mb-4">
              or use your email and password
            </span>
            <input
              type="email"
              placeholder="E-Mail"
              className="w-full p-3 mb-4 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <input
              type="password"
              placeholder="Password"
              className="w-full p-3 mb-4 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <p className="text-sm mb-4">
              <a href="#" className="text-[#006953] hover:text-green-700">
                Forgot Your Password?
              </a>
            </p>
            <button
              type="button"
              className="bg-[#006953] text-white py-2 px-8 rounded-lg uppercase font-semibold text-sm tracking-wide hover:bg-green-700 transition-all"
            >
              Sign In
            </button>
          </form>
        </div>

        {/* Sign Up Form */}
        <div
          className={`form-container w-1/2 flex flex-col justify-center items-center px-10 bg-white transition-transform duration-700 ${
            isActive ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <form className="flex flex-col justify-center items-center w-full">
            <h1 className="text-3xl font-bold mb-4">Create Account</h1>
            <div className="flex justify-center mb-4">
              <a
                href="#"
                className="flex items-center justify-center w-10 h-10 border border-gray-400 rounded-full mx-2 hover:bg-[#006953] hover:text-white transition-all"
              >
                <FontAwesomeIcon icon={faGoogle} />
              </a>
              <a
                href="#"
                className="flex items-center justify-center w-10 h-10 border border-gray-400 rounded-full mx-2 hover:bg-[#006953] hover:text-white transition-all"
              >
                <FontAwesomeIcon icon={faFacebookF} />
              </a>
            </div>
            <span className="text-sm text-gray-600 mb-4">
              or use your email for registration
            </span>
            <input
              type="text"
              placeholder="Name"
              className="w-full p-3 mb-4 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <input
              type="email"
              placeholder="E-Mail"
              className="w-full p-3 mb-4 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <input
              type="password"
              placeholder="Password"
              className="w-full p-3 mb-6 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <input
              type="number"
              placeholder="Mobile No."
              className="w-full p-3 mb-4 bg-gray-100 rounded-lg outline-none focus:ring-2 focus:ring-[#006953] transition-all"
            />
            <button
              type="button"
              onClick={handleSignUp}  // Trigger redirect on sign up
              className="bg-[#006953] text-white py-2 px-8 rounded-lg uppercase font-semibold text-sm tracking-wide hover:bg-green-700 transition-all"
            >
              Sign Up
            </button>
          </form>
        </div>

        {/* Toggle Panels */}
        <div
          className={`absolute top-0 right-0 w-1/2 h-full bg-[#006953] text-white flex items-center justify-center transition-transform duration-700 ${
            isActive ? "translate-x-[-100%]" : "translate-x-0"
          }`}
        >
          <div className="flex flex-col items-center justify-center p-12 text-center">
            <h1 className="text-3xl font-bold mb-4 text-white">
              {isActive ? "Hello!" : "Welcome Back!"}
            </h1>
            <p className="mb-6">
              {isActive
                ? "Register now with your personal details."
                : "Enter your details to sign in and continue."}
            </p>
            <button
              className="bg-transparent border border-white text-white py-2 px-8 rounded-lg uppercase font-semibold text-sm tracking-wide hover:bg-white hover:text-[#006953] transition-all"
              onClick={isActive ? handleLoginClick : handleRegisterClick}
            >
              {isActive ? "Sign In" : "Sign Up"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
