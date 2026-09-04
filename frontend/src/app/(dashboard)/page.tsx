
export default function DashboardHome() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center">
      <div className="w-16 h-16 bg-[#1A1A1A] rounded-2xl flex items-center justify-center mb-6 border border-gray-800">
        <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center font-bold">
          AI
        </div>
      </div>
      <h1 className="text-2xl font-bold mb-2">How can I help you today?</h1>
      <p className="text-gray-500 mb-8 max-w-md text-center">Select a conversation from the sidebar or start a new chat to begin your research.</p>
    </div>
  );
}

