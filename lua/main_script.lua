luanet.load_assembly "System"

local server = {}

---- TCP CONFIGURATION
local Encoder = luanet.import_type "System.Text.Encoding"
local TcpClient = luanet.import_type("System.Net.Sockets.TcpClient")
local IO = luanet.import_type "System.IO.Path"
server.USE_CLIPBOARD = true
server.SCREENSHOT_FILE = ""

function server.init(port, use_clipboard, file_name)
  server.mySocket = TcpClient("localhost", port)
  server.stream = server.mySocket:GetStream()
  server.sMessage = "myMessage"
  server.rMessage = "receivedData"
  server.steering_action = 0
  server.USE_CLIPBOARD = use_clipboard
  server.SCREENSHOT_FILE = server.getTMPDir() .. file_name
end

function server.sendMsg()
  local message = Encoder.UTF8:GetBytes(server.sMessage)
  server.stream:Write(message, 0, string.len(server.sMessage))
  --console.log("sent: " .. sMessage)
end

function server.recvData()
  local buffer = "000000000000000000000000000000000" -- we should actually just receive a value to steer
  local byteBuffer = Encoder.UTF8:GetBytes(buffer)
  local bytesSent = server.stream:Read(byteBuffer, 0, string.len(buffer))
  local receivedMessage = Encoder.UTF8:GetString(byteBuffer)
  return string.sub(receivedMessage, 1, bytesSent)
end

function server.getTMPDir()
  return IO.GetTempPath()
end

BOX_CENTER_X, BOX_CENTER_Y = 160, 215
BOX_WIDTH, BOX_HEIGHT = 100, 4
SLIDER_WIDTH, SLIDER_HIEGHT = 4, 16
function server.draw_info()
  gui.drawBox(BOX_CENTER_X - BOX_WIDTH / 2, BOX_CENTER_Y - BOX_HEIGHT / 2,
              BOX_CENTER_X + BOX_WIDTH / 2, BOX_CENTER_Y + BOX_HEIGHT / 2,
              none, 0x60FFFFFF)
  gui.drawBox(BOX_CENTER_X + server.steering_action*(BOX_WIDTH / 2) - SLIDER_WIDTH / 2, BOX_CENTER_Y - SLIDER_HIEGHT / 2,
              BOX_CENTER_X + server.steering_action*(BOX_WIDTH / 2) + SLIDER_WIDTH / 2, BOX_CENTER_Y + SLIDER_HIEGHT / 2,
              none, 0xFFFF0000)
end

---- END TCP CONFIGURATION

---- SOME OTHER GLOBALS

--[[ How many frames to wait before sending a new prediction request. If you're using a file, you
may want to consider adding some frames here. ]]--
local WAIT_FRAMES = 5

savestate.loadslot(2)
savestate.saveslot(2) -- save current slot for reset purposes
local util = require("util")


--- reinforcement variables
local new_progress = util.readProgress()
local old_progress = 0
local reward = 0
local done = "False"
local predictions = 0
local totalReward = 0

function server.request_prediction()
  predictions = predictions + 1
  new_progress = util.readProgress()
  reward = new_progress - old_progress
  old_progress = new_progress
  if reward > 0 then
    reward = reward
  else
    reward = -.1
  end
  totalReward = totalReward + reward
  if (-totalReward) > 100 then
    done = "True"
    totalReward = 0
  else
    done = "False"
  end
  --console.log(totalReward)

  --console.log(reward)
  if server.USE_CLIPBOARD then
    client.screenshottoclipboard()
    server.sMessage = "MESSAGE screenshot_clip_reward_" .. reward .. "_done_" .. done .. "\n"
  else
    client.screenshot(server.SCREENSHOT_FILE)
    server.sMessage = "MESSAGE screenshot_" .. server.SCREENSHOT_FILE .. "_reward_" .. reward .. "_done_" .. done .. "\n"
    --outgoing_message = "PREDICT:" .. SCREENSHOT_FILE .. "\n"
  end
end

function server.start()
    while util.readProgress() < 3 do -- 3 means 3 laps
      -- Process the outgoing message.
      server.request_prediction()
      server.sendMsg()
      --- Process incoming message
      server.rMessage = server.recvData()

      if string.find(server.rMessage, "RESET") ~= nil then
        console.log('Reset game - LOADING SLOT 2 Which we saved at the beginning')
        savestate.loadslot(2)
        client.unpause()
      elseif string.find(server.rMessage, "PREDICTIONERROR") == nil then
        --console.log('current message: ' .. rMessage)
        local a1, a2 = string.match(server.rMessage, "(-?%d.%d*)%s*(-?%d.%d*)")
        --console.log('a1: ' .. a1 .. ' a2: ' .. a2)
        local acceleration = true
        if a1 == nil then
          a1 = server.rMessage -- if we only have 1 action
        else
          -- we have 2 actions
          if 0 < tonumber(a2) then
            acceleration = true
          else
            acceleration = false
          end
        end
        server.steering_action = tonumber(a1)


        --console.log('current message: ' .. rMessage)
        --console.log('current steering_action: ' .. steering_action)

        for i=1, WAIT_FRAMES do
          --console.log('wait frame')
          joypad.set({["P1 A"] = acceleration})
          joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(server.steering_action) })
          server.draw_info()
          emu.frameadvance()
        end
      else
        print("Prediction error...")
      end

      if util.readProgress() > 3.0 then
        console.log('Reset game - LOADING SLOT 2 Which we saved at the beginning')
        savestate.loadslot(2)
        client.unpause()
      end

    end

end

return server

